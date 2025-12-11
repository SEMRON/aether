from __future__ import annotations

from dataclasses import dataclass, field
from nicegui import ui
from nicegui.events import UploadEventArguments
from pathlib import Path
from typing import Any, Optional, List, Callable
from typing import Callable, Optional
import argparse
import asyncio
import json
import re
import tempfile
import traceback
from datetime import datetime

from distqat.setup.create_setup_files import *
from .ssh_runner import run_bash_script_via_ssh, SSHSession
from .config import Server

@dataclass
class GuiValidationError:
    context: str  # e.g. "configuration against machine info"
    message: str  # the individual error text

@dataclass
class GuiValidationReport:
    """Holds validation errors and related log text."""
    errors: List[GuiValidationError] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

    def no_errors(self) -> bool:
        return not self.errors

@dataclass
class GuiSetupResult:
    """Return type of the GUI orchestration logic."""
    success: bool
    cfg: Optional[Config]
    machine_info: Optional[dict[str, Any]]
    target: Optional[ExportTarget]
    profile: Optional[SetupProfile]
    output_text: Optional[str]
    output_path: Optional[Path]
    logs: List[str]
    validation: GuiValidationReport

@dataclass
class SetupTabHandle:
    preset_simplified: Callable[[str, str, Optional[str], Optional[str], Optional[str]], None]
    set_ssh_server: Callable[[Optional[Server]], None]

_gui_available_targets = set(t for t in ExportTarget if t not in [ExportTarget.SKY_LAUNCH, ExportTarget.DIR_WITH_RUNNER])

def _validate_with_report(
    errors: List[str],
    validation_type: str,
    obj_to_dump: Any | None,
    report: GuiValidationReport,
) -> GuiValidationReport:
    """GUI version of report_validation_errors_and_exit: collects info instead of exiting."""
    if not errors:
        return report

    report.logs.append(f"An error occurred while validating {validation_type}:")
    if obj_to_dump is not None:
        try:
            report.logs.append(json.dumps(obj_to_dump, indent=2))
        except Exception:
            report.logs.append(f"<failed to serialize {validation_type}>")

    report.logs.append("\nErrors:\n" + "=" * 80)

    for i, e in enumerate(errors, start=1):
        report.logs.append(f"Error {i}:\n    " + "\n    ".join(e.splitlines()))
        report.errors.append(GuiValidationError(context=validation_type, message=e))

    return report

def _save_generated_output(text: str, syntax: str, path: Optional[Path]) -> Optional[Path]:
    """GUI-friendly variant of save_or_print_formatted: writes or does nothing, no exit()."""
    if not path:
        return None

    syntax_to_valid_exts = {
        "bash": {".sh"},
        "sh": {".sh"},
        "yaml": {".yaml", ".yml"},
        "yml": {".yaml", ".yml"},
        "json": {".json"},
    }

    valid_exts = syntax_to_valid_exts.get(syntax.lower(), set())

    if path.suffix != "":
        if valid_exts and path.suffix not in valid_exts:
            raise ValueError(
                f"Path '{path}' has extension '{path.suffix}' which doesn't match syntax '{syntax}'. "
                f"Expected one of: {', '.join(sorted(valid_exts))} or no extension."
            )

    try:
        path.write_text(text)
    except Exception as e:
        raise IOError(f"Failed to save to {path}: {e}") from e

    return path

def get_setup_configs() -> dict[str, Callable[[], str]]:
    """
    Returns a dict mapping config names to lambdas that load their contents.
    Configs are discovered by listing the setup_configs/ directory.
    """
    configs = {}
    setup_configs_dir = Path("setup_configs")

    if not setup_configs_dir.exists():
        return configs

    for config_file in setup_configs_dir.glob("*.json"):
        config_name = config_file.stem
        # Create a closure to capture the specific file path
        configs[config_name] = lambda path=config_file: path.read_text()

    if not configs:
        print(f"No sample config files found in {setup_configs_dir}", file=sys.stderr)

    return configs

def run_setup_from_gui(
    *,
    # core selection
    config_path: Optional[Path],
    machine_info_file: Optional[Path],
    for_this_machine: bool,
    dump_config: bool,
    target_name: Optional[str],
    profile_name: str,
    # repo
    source_url: Optional[str],
    source_ref: Optional[str],
    sources_target_dir: Optional[str],
    # output
    out_path: Optional[Path],
    # runtime
    commands: Optional[List[str]],
    # ssh / keys
    clone_key_path: Optional[Path],
    authorized_key_file: Optional[Path],
    authorized_keys: List[str],
    forward_authorized_keys: bool,
    local_private_key: Optional[Path],
    # users / GPU
    user: Optional[str],
    management_user: Optional[str],
    gpu_vendor_name: Optional[str],
    install_drivers: Optional[bool],
) -> GuiSetupResult:
    """
    Re-implementation of main() logic for GUI usage.
    Raises on fatal errors; returns logs and output text for display.
    """

    logs: List[str] = []
    validation = GuiValidationReport()

    # GPU vendor lookup
    gpu_vendor = None
    if gpu_vendor_name:
        gpu_vendor = next((v for v in GPU_VENDOR if v.vendor_name == gpu_vendor_name), None)
        if gpu_vendor is None:
            raise ValueError(f"Unknown GPU vendor '{gpu_vendor_name}'.")

    # Resolve machine info
    machine_info: Optional[dict[str, Any]] = None
    if for_this_machine:
        logs.append("Obtaining machine information from local system...")
        machine_info = get_local_machine_info()
    elif machine_info_file or not config_path:
        if machine_info_file:
            logs.append(f"Loading machine info from file: {machine_info_file}")
            with open(machine_info_file, "r") as f:
                machine_info = json.load(f)
        else:
            raise ValueError(
                "Machine info is required (no config file provided). "
                "Provide either a machine info file or JSON text."
            )

    # Build config
    if config_path:
        logs.append(f"Loading config from file: {config_path}")
        with open(config_path, "r") as f:
            config_data = json.load(f)
            cfg = Config.from_dict(config_data)

        # Resolve source_url: args -> config -> upstream
        if source_url:
            effective_source_url = source_url
        elif cfg.source_url:
            effective_source_url = cfg.source_url
        else:
            upstream_url, _ = get_upstream_repo_url_and_ref_for_pwd(
                private_key_path=clone_key_path
            )
            effective_source_url = upstream_url
        logs.append(f"Using source URL: {effective_source_url}")

        # Resolve source_ref: args -> config -> upstream
        if source_ref:
            effective_source_ref = source_ref
        elif cfg.source_ref:
            effective_source_ref = cfg.source_ref
        else:
            _, upstream_ref = get_upstream_repo_url_and_ref_for_pwd(
                private_key_path=clone_key_path
            )
            effective_source_ref = upstream_ref
        logs.append(f"Using source ref: {effective_source_ref}")

        # Prepare authorized keys: reuse helper via a Namespace-like object
        ak_args = argparse.Namespace(
            authorized_key_file=authorized_key_file,
            forward_authorized_keys=forward_authorized_keys,
            local_private_key=local_private_key,
            authorized_keys=authorized_keys,
        )
        node_login_authorized_keys = load_authorized_keys(ak_args)

        effective_sources_target_dir = (
            sources_target_dir
            if sources_target_dir
            else get_repo_name_from_git_url(cfg.source_url or effective_source_url)
        )

        cfg = update_config(
            cfg=cfg,
            clone_key_path=clone_key_path,
            node_login_authorized_keys=node_login_authorized_keys,
            source_url=effective_source_url,
            source_ref=effective_source_ref,
            commands=commands,
            sources_target_dir=effective_sources_target_dir,
            gpu_vendor=gpu_vendor,
            user_override=user,
            install_drivers=install_drivers,
            management_user=management_user,
        )

    elif machine_info:
        # Validate machine info
        validation = _validate_with_report(
            machine_info_errors(machine_info),
            "the user provided machine info",
            machine_info,
            report=validation,
        )

        if not source_url:
            logs.append("Determining upstream repo URL/ref from current working directory...")
            effective_source_url, effective_source_ref = get_upstream_repo_url_and_ref_for_pwd(
                private_key_path=clone_key_path
            )
        else:
            effective_source_url = source_url
            effective_source_ref = None

        if source_ref:
            effective_source_ref = source_ref

        logs.append(f"Using source URL: {effective_source_url}")
        logs.append(f"Using source ref: {effective_source_ref}")

        ak_args = argparse.Namespace(
            authorized_key_file=authorized_key_file,
            forward_authorized_keys=forward_authorized_keys,
            local_private_key=local_private_key,
        )
        node_login_authorized_keys = load_authorized_keys(ak_args)

        effective_sources_target_dir = (
            sources_target_dir
            if sources_target_dir
            else get_repo_name_from_git_url(effective_source_url)
        )

        cfg = load_config(
            machine_info=machine_info,
            clone_key_path=clone_key_path,
            node_login_authorized_keys=node_login_authorized_keys,
            source_url=effective_source_url,
            source_ref=effective_source_ref,
            commands=commands,
            sources_target_dir=effective_sources_target_dir,
            gpu_vendor=gpu_vendor,
            user_override=user,
            install_drivers=install_drivers,
            management_user=management_user,
        )
    else:
        raise RuntimeError("BUG: unreachable configuration branch (no config and no machine_info).")

    # Validate configuration against machine info
    if machine_info:
        validation = _validate_with_report(
            config_against_machine_info_errors(cfg, machine_info),
            "the configuration against machine info",
            {"config": cfg.to_dict(), "machine_info": machine_info},
            report=validation,
        )

    # Validate configuration alone
    validation = _validate_with_report(
        config_errors(cfg),
        "the generated configuration",
        cfg.to_dict(),
        report=validation,
    )

    # Module matrix
    mods = [x() for x in allSetupModuleClasses]
    from io import StringIO

    buf = StringIO()
    print_matrix_table(
        row_items=mods,
        col_items=list(SetupProfile) + [None] + list(_gui_available_targets),
        check_func=lambda mod, tgt_or_profile: (
            "-" if not mod.enabled(cfg) else (
                "P" if (tgt_or_profile in mod.profiles()) else
                "." if isinstance(tgt_or_profile, SetupProfile) else
                tgt_or_profile in mod.targets() if isinstance(tgt_or_profile, ExportTarget) else
                False
            )
        ),
        row_label="Module ↓ | Profiles/Targets →",
        file=buf,
    )
    validation.logs.append(buf.getvalue())

    # Dump config-only mode
    if dump_config:
        return GuiSetupResult(
            success=True,
            cfg=cfg,
            machine_info=machine_info,
            target=None,
            profile=None,
            output_text=None,
            output_path=None,
            logs=logs,
            validation=validation,
        )

    # Non-dump mode: need a target and profile
    target = None
    profile = None
    error_msgs = []

    if not target_name:
        error_msgs.append("Target must be selected when not dumping config.")
        success = False
    else:
        try:
            target = ExportTarget[target_name.upper()]
        except KeyError as e:
            error_msgs.append(f"Unknown target '{target_name}'.")

        if not error_msgs:
            try:
                profile = SetupProfile[profile_name.upper()]
            except KeyError as e:
                error_msgs.append(f"Unknown profile '{profile_name}'.")

    active = [
        m for m in mods
        if (target in set(m.targets()))
        and (profile in set(m.profiles()))
        and m.enabled(cfg)
    ]

    output_text = ""
    saved_path: Optional[Path] = None

    if validation.no_errors():
        try:
            if target is ExportTarget.CLOUD_INIT:
                output_text = emit_cloud_init(cfg, active)
                saved_path = _save_generated_output(output_text, syntax="yaml", path=out_path)
            elif target is ExportTarget.BASH_SCRIPT:
                output_text = emit_bash_script(cfg, active)
                saved_path = _save_generated_output(output_text, syntax="bash", path=out_path)
            elif target is ExportTarget.DIR_WITH_RUNNER:
                if not out_path:
                    error_msgs.append("--out directory is required for DIR_WITH_RUNNER target.")
                else:
                    emit_dir_with_runner(cfg, active, out_path)
                    output_text = f"Wrote directory: {out_path}"
                    saved_path = out_path
            elif target is ExportTarget.SKY_LAUNCH:
                output_text = emit_skypilot_launch(cfg, active)
                saved_path = _save_generated_output(output_text, syntax="yaml", path=out_path)
            else:
                error_msgs.append(f"Unknown target '{target}'.")
        except TemplateRenderError as ex:
            context_str = json.dumps(ex.context, indent=2)
            msg = (
                f"Error rendering template '{ex.template_file}'.\n"
                f"Context:\n{context_str}\n\n"
                f"{ex}"
            )
            logs.append(msg)

    if error_msgs:
        logs.extend(f"Error: {error_msg}" for error_msg in error_msgs)

    for error_msg in error_msgs:
        validation.errors.append(GuiValidationError(
            context="target output generation",
            message=error_msg
        ))

    success = validation.no_errors()

    if success and saved_path:
        logs.append(f"Output written to: {saved_path}")
    elif success:
        logs.append("Output generated (not saved to a file).")

    return GuiSetupResult(
        success=success,
        cfg=cfg,
        machine_info=machine_info,
        target=target,
        profile=profile,
        output_text=output_text,
        output_path=saved_path,
        logs=logs,
        validation=validation,
    )


def _style_wide(component) -> Any:
    """Ensure all text inputs / dropdowns are at least 400px wide."""
    return component.style("min-width: 400px")

@dataclass
class JsonSourceUI:
    """Handles a JSON source provided via file upload or pasted text."""
    status_label: ui.label
    preview: ui.code
    file_dialog_button: Optional[ui.upload]
    paste_button: Optional[ui.button]
    sample_select: Optional[ui.select]

    get_file_path: Callable[[], Optional[Path]]
    get_text: Callable[[], Optional[str]]
    set_simplified: Callable[[bool], None]

    select_sample: Callable[[Optional[str]], None]


def create_json_source_ui(
    *,
    choose_label: str,
    paste_button_label: str,
    paste_dialog_title: str,
    status_initial: str,
    temp_prefix: str,
    temp_suffix: str = ".json",
    sample_files: Optional[dict[str, Callable[[], str]]] = None,
    sample_files_label: Optional[str] = None,
    on_value_changed: Optional[Callable[[], None]] = None,
) -> JsonSourceUI:
    """
    Create a JSON file-or-paste input with validation and preview.

    In simplified mode, only the sample selector is shown (if available),
    not the upload and paste options.
    """

    json_file_path: Optional[Path] = None
    json_text_value: Optional[str] = None

    status_label = ui.label(status_initial).classes("text-caption text-grey-7")
    preview = ui.code("", language="json").classes("w-full")

    def _parse_and_store_json(text: str, source: str) -> None:
        nonlocal json_file_path, json_text_value

        try:
            parsed = json.loads(text)
            pretty = json.dumps(parsed, indent=2)
        except json.JSONDecodeError as e:
            ui.notify(f"JSON is not valid: {e}", type="negative")
            return

        # store pretty JSON as text
        json_text_value = pretty

        # store in a temp file
        _, tmp_name = tempfile.mkstemp(suffix=temp_suffix, prefix=temp_prefix)
        Path(tmp_name).write_text(pretty)
        json_file_path = Path(tmp_name)

        # update UI
        status_label.text = f"Loaded from {source}"
        preview.set_content(pretty)

        # Call the callback if provided
        if on_value_changed:
            on_value_changed()

    # container for upload + paste (can be hidden in simplified mode)
    advanced_container = ui.column().classes("w-full")
    with advanced_container:
        # --- upload button ---
        async def _on_upload(e: UploadEventArguments) -> None:
            try:
                text = await e.file.text()  # returns str
            except Exception as ex:
                ui.notify(f"Failed to read uploaded file: {ex}", type="negative")
                return
            _parse_and_store_json(text, f"file '{e.file.name}'")

        file_upload_widget = ui.upload(
            label=choose_label,
            auto_upload=True,
            multiple=False,
            on_upload=_on_upload,
        )

        ui.label("OR").classes("text-body2 text-grey-7")

        # --- paste dialog ---
        paste_dialog = ui.dialog()
        with paste_dialog, ui.card().classes("w-[600px]"):
            ui.label(paste_dialog_title).classes("text-subtitle1")
            paste_textarea = ui.textarea().props("rows=10").classes("w-full")
            with ui.row().classes("justify-end q-gutter-sm q-mt-sm"):
                ui.button("Cancel", on_click=paste_dialog.close)

                def _confirm_paste() -> None:
                    text = paste_textarea.value or ""
                    _parse_and_store_json(text, "pasted contents")
                    paste_dialog.close()

                ui.button("Use", on_click=_confirm_paste)

        paste_button = ui.button(paste_button_label, on_click=paste_dialog.open)

    # --- sample files selector (always visible if present, even in simplified mode) ---
    sample_select = None
    sample_or_label = None
    if sample_files:
        sample_or_label = ui.label("OR").classes("text-body2 text-grey-7")
        ui.label("Select:")

        def _load_sample() -> None:
            selected = sample_select.value if sample_select else None
            if selected and selected in sample_files:
                try:
                    text = sample_files[selected]()
                    _parse_and_store_json(text, f"sample '{selected}'")
                except Exception as ex:
                    ui.notify(f"Failed to load sample '{selected}': {ex}", type="negative")

        sample_select = _style_wide(
            ui.select(
                options=list(sample_files.keys()),
                label=sample_files_label or "Load sample",
                with_input=False,
                on_change=_load_sample,
            ).props("clearable")
        )

    def _get_file_path() -> Optional[Path]:
        return json_file_path

    def _get_text() -> Optional[str]:
        return json_text_value

    def _set_simplified(flag: bool) -> None:
        # In simplified mode, hide upload/paste, keep only the sample selector.
        advanced_container.visible = not flag
        if sample_or_label is not None:
            # If there is no upload/paste visible, hide the preceding "OR".
            sample_or_label.visible = not flag

    def _select_sample(name: Optional[str]) -> None:
        if not sample_files or not sample_select:
            return
        if not name:
            sample_select.value = None
            return
        if name in sample_files:
            # set the select, then reuse the existing loader
            sample_select.value = name
            _load_sample()

    return JsonSourceUI(
        status_label=status_label,
        preview=preview,
        file_dialog_button=file_upload_widget if 'file_upload_widget' in locals() else None,
        paste_button=paste_button if 'paste_button' in locals() else None,
        sample_select=sample_select,
        get_file_path=_get_file_path,
        get_text=_get_text,
        set_simplified=_set_simplified,
        select_sample=_select_sample,
    )

def create_setup_tab(tab) -> SetupTabHandle:
    """NiceGUI UI: expose setup functionality as a single structured tab."""

    # local state for config and machine info inputs
    last_output_text: str = ""
    last_output_filename: str = "setup_output.txt"

    # SSH-related state for remote execution of the generated bash script
    ssh_server: Optional[Server] = None
    ssh_task: Optional[asyncio.Task] = None

    # UI blocking state
    ui_blocked_by_preset: bool = False
    ui_blocked_by_ssh: bool = False

    def reset_output(x = None, from_sample_change: bool = False):
        """
        Reset the output, to be called when setting a new configuration.
        This will reset the visibility of the "Generated script", configuration and remote console output.
        
        Args:
            x: Unused parameter for compatibility with event handlers
            from_sample_change: If True, this reset is from changing the sample config
        """
        nonlocal last_output_text, last_output_filename

        # Check if UI is blocked
        # Allow sample changes during preset lock but not during SSH
        if ui_blocked_by_ssh:
            ui.notify("Configuration is locked while SSH task is running. Wait for completion or click 'Override' to unlock.", type="warning")
            return
        if ui_blocked_by_preset and not from_sample_change:
            ui.notify("Configuration is locked. Click 'Unlock Configuration' to make changes.", type="warning")
            return

        ui.notify("Resetting output")

        # Clear output-related state
        last_output_text = ""
        last_output_filename = "setup_output.txt"

        # Hide output-related UI elements (only if they exist)
        try:
            output_expansion.visible = False
            output_expansion.value = True  # Default to expanded when it becomes visible
        except NameError:
            pass

        try:
            config_display_card.visible = False
        except NameError:
            pass

        try:
            remote_exec_card.visible = False
        except NameError:
            pass

        # Clear output content (only if they exist)
        try:
            output_textarea.value = ""
        except NameError:
            pass

        try:
            config_display.set_content("")
        except NameError:
            pass

        # Clear remote execution log (only if they exist)
        try:
            remote_log_container.clear()
        except NameError:
            pass

        try:
            remote_status_label.text = "Idle"
        except NameError:
            pass

        # Reset main status (only if it exists)
        try:
            status_label.text = "Idle"
        except NameError:
            pass

        # Hide validation elements (only if they exist)
        try:
            validation_warning_label.visible = False
        except NameError:
            pass

        try:
            details_expansion.visible = False
            details_expansion.value = False
        except NameError:
            pass

        # Clear validation content (only if they exist)
        try:
            validation_table.rows = []
        except NameError:
            pass

        try:
            log_output.value = ""
        except NameError:
            pass

        try:
            preset_info_card.visible = False
        except NameError:
            pass

        try:
            ssh_block_card.visible = False
        except NameError:
            pass

        try:
            log_file_label.visible = False
            log_file_label.text = ""
        except NameError:
            pass

    def set_config_inputs_enabled(enabled: bool, keep_sample_select_enabled: bool = False):
        """Enable or disable configuration input elements (but not run buttons).
        
        Args:
            enabled: Whether to enable/disable the inputs
            keep_sample_select_enabled: If True, sample selectors remain enabled regardless
        """
        # Toggle state for all input fields
        mode_toggle.enabled = enabled
        if config_json_ui.file_dialog_button:
            config_json_ui.file_dialog_button.enabled = enabled
        if config_json_ui.paste_button:
            config_json_ui.paste_button.enabled = enabled
        if config_json_ui.sample_select:
            # Sample select can optionally remain enabled
            config_json_ui.sample_select.enabled = enabled or keep_sample_select_enabled
        if machine_info_json_ui.file_dialog_button:
            machine_info_json_ui.file_dialog_button.enabled = enabled
        if machine_info_json_ui.paste_button:
            machine_info_json_ui.paste_button.enabled = enabled
        if machine_info_json_ui.sample_select:
            machine_info_json_ui.sample_select.enabled = enabled
        for_this_machine_checkbox.enabled = enabled
        source_url_input.enabled = enabled
        source_ref_input.enabled = enabled
        sources_target_dir_input.enabled = enabled
        clone_key_input.enabled = enabled
        authorized_key_file_input.enabled = enabled
        forward_authorized_checkbox.enabled = enabled
        local_private_key_input.enabled = enabled
        authorized_keys_textarea.enabled = enabled
        user_input.enabled = enabled
        management_user_input.enabled = enabled
        gpu_vendor_select.enabled = enabled
        install_drivers_select.enabled = enabled
        commands_textarea.enabled = enabled
        target_select.enabled = enabled
        profile_select.enabled = enabled
        dump_config_checkbox.enabled = enabled
    
    def set_run_buttons_enabled(enabled: bool):
        """Enable or disable run buttons."""
        run_button.enabled = enabled
        try:
            remote_run_button.enabled = enabled
        except NameError:
            pass

    def unlock_from_preset():
        """Unlock UI after preset configuration."""
        nonlocal ui_blocked_by_preset
        ui_blocked_by_preset = False
        preset_info_card.visible = False
        set_config_inputs_enabled(True)
        # Run buttons should remain enabled unless SSH is running
        if not ui_blocked_by_ssh:
            set_run_buttons_enabled(True)
        ui.notify("Configuration unlocked. You can now modify the values.", type="positive")

    def set_ssh_block(blocked: bool):
        """Set SSH blocking state explicitly."""
        nonlocal ui_blocked_by_ssh
        if blocked:
            ui_blocked_by_ssh = True
            ssh_block_card.visible = True
            # Block everything during SSH execution (including sample selectors)
            set_config_inputs_enabled(False, keep_sample_select_enabled=False)
            set_run_buttons_enabled(False)
        else:
            ui_blocked_by_ssh = False
            ssh_block_card.visible = False
            # Restore state based on preset lock
            if not ui_blocked_by_preset:
                set_config_inputs_enabled(True)
            else:
                # If preset locked, keep sample selector enabled
                set_config_inputs_enabled(False, keep_sample_select_enabled=True)
            set_run_buttons_enabled(True)  # Run buttons always enabled when SSH not running

    def override_ssh_block():
        """Override SSH task blocking."""
        nonlocal ui_blocked_by_ssh
        ui_blocked_by_ssh = False
        ssh_block_card.visible = False
        if not ui_blocked_by_preset:
            set_config_inputs_enabled(True)
        else:
            # If preset locked, keep sample selector enabled
            set_config_inputs_enabled(False, keep_sample_select_enabled=True)
        set_run_buttons_enabled(True)  # Run buttons always enabled when SSH not running
        ui.notify("SSH task lock overridden. Configuration is now editable.", type="info")

    with ui.tab_panel(tab):
        # simplified / full mode toggle (simplified by default)
        mode_toggle = ui.toggle(
            {'simplified': 'Simplified', 'full': 'Full'},
            value='simplified',
        ).classes("q-mb-md")

        def is_simplified() -> bool:
            return mode_toggle.value == 'simplified'

        ui.label("System setup configuration").classes("text-h5")
        
        with ui.expansion("Configuration", icon="settings").classes("w-full mb-4"):
            with ui.column().classes("w-full gap-2 p-1"):
                main_desc_label = ui.label(
                    "Generate setup scripts or configuration for target machines from either a full config "
                    "JSON or machine info JSON. This is a GUI wrapper around the same core logic as the "
                    "command-line tool."
                ).classes("text-body2 text-grey-7")

                # Info card for simplified mode
                preset_info_card = ui.card().classes("w-full q-mt-md border-l-4 border-blue-500")
                with preset_info_card:
                    with ui.row().classes("w-full items-start"):
                        with ui.column().classes("flex-grow"):
                            ui.label("ℹ️ Auto-configured from selected server").classes("text-subtitle1 text-blue-700")
                            ui.label(
                                "The information in this UI was automatically updated to match the server you selected. "
                                "The configuration is currently locked to prevent accidental changes. "
                                "(assuming you set the information correctly in the \"SERVERS\" tab)"
                            ).classes("text-body2 text-grey-7")
                            ui.label(
                                "Proceed straight to the bottom to generate the server setup script, and run it on the server."
                            ).classes("text-body2 font-weight-medium q-mt-sm")
                        ui.button("Unlock Configuration", on_click=unlock_from_preset, icon="lock_open").props("flat")
                preset_info_card.visible = False

                # SSH task blocking card
                ssh_block_card = ui.card().classes("w-full q-mt-md border-l-4 border-orange-500")
                with ssh_block_card:
                    with ui.row().classes("w-full items-center"):
                        ui.spinner(size='sm').classes("q-mr-sm")
                        with ui.column().classes("flex-grow"):
                            ui.label("⚠️ SSH task is running").classes("text-subtitle1 text-orange-700")
                            ui.label(
                                "Configuration is locked while an SSH task is executing. "
                                "Please wait for it to complete or click 'Override' to unlock."
                            ).classes("text-body2 text-grey-7")
                        ui.button("Override", on_click=override_ssh_block, icon="block").props("flat")
                ssh_block_card.visible = False

                # ------------------------------------------------------------------
                # Panel 1: Base configuration / machine info
                # ------------------------------------------------------------------
                base_config_card = ui.card().classes("w-full q-mt-md")
                with base_config_card:
                    base_title_label = ui.label("Lorem Ipsum").classes("text-subtitle1")
                    base_desc_label = ui.label("Lorem Ipsum").classes("text-body2 text-grey-7")

                    # -------------------- Config JSON ------------------------------
                    config_title_label = ui.label("Lorem Ipsum").classes("text-body1 q-mt-md")
                    config_help_label = ui.label("Lorem Ipsum").classes("text-body2 text-grey-7")

                    with ui.row().classes("items-center q-gutter-md q-mt-sm"):
                        config_json_ui = create_json_source_ui(
                            choose_label="Choose config JSON file",
                            paste_button_label="Paste config JSON",
                            paste_dialog_title="Paste config JSON",
                            status_initial="No config loaded yet.",
                            temp_prefix="config_",
                            sample_files=get_setup_configs(),
                            sample_files_label="Sample Configs",
                            on_value_changed=lambda: reset_output(from_sample_change=True) if not ui_blocked_by_ssh else None
                        )

                    # -------------------- Machine info JSON -------------------------
                    and_or_row = ui.row().classes("q-my-md items-center w-full items-center")
                    with and_or_row:
                        ui.separator()
                        ui.label("AND / OR").classes("text-body2 text-grey-7 px-2")
                        ui.separator()

                    machine_info_section = ui.column().classes("w-full")
                    with machine_info_section:
                        ui.label("Machine info JSON (optional)").classes("text-body1")
                        # -------------------- Reference script --------------------------
                        with ui.card().classes("w-full q-mt-md border-l-4 border-gray-400"):
                            ui.label("Reference machine info script").classes("text-body1")
                            ui.label(
                                "This is a bash script you can run in any shell on a target machine.\n"
                                "It automatically gets the target machine hardware and software configuration, and generates a \"machine info\" json string.\n"
                                "You can then insert that \"machine info\" below, to auto configure many aspects of the installation.\n"
                                "(or replecate e.g. the user, in case of generating a cloud_init)"
                            ).classes("text-body2 text-grey-7")

                            try:
                                reference_script_text = GET_MACHINE_INFO_SCRIPT.read_text()
                            except Exception as ex:
                                reference_script_text = f"# Failed to read script: {ex}"

                            script_dialog = ui.dialog()

                            with script_dialog, ui.card().classes("w-[800px]"):
                                ui.label("get-initial-machine-info.sh").classes("text-subtitle1")
                                ui.code(reference_script_text, language="bash").classes("w-full")
                                with ui.row().classes("justify-end q-gutter-sm q-mt-sm"):
                                    ui.button("Close", on_click=script_dialog.close)

                            with ui.row().classes("q-mt-xs q-gutter-sm"):
                                def _copy_script_to_clipboard() -> None:
                                    ui.run_javascript(
                                        f"navigator.clipboard.writeText({json.dumps(reference_script_text)})"
                                    )
                                    ui.notify("Script copied to clipboard.", type="positive")

                                ui.button("Copy script to clipboard", on_click=_copy_script_to_clipboard)
                                ui.button("View script", on_click=script_dialog.open)

                        ui.label(
                            "Machine info JSON is obtained by running the reference machine info script on the "
                            "target machine. It is used to infer a matching config when no full config is given."
                        ).classes("text-body2 text-grey-7")


                        with ui.row().classes("items-center q-gutter-md q-mt-sm"):
                            machine_info_json_ui = create_json_source_ui(
                                choose_label="Choose machine info JSON file",
                                paste_button_label="Paste machine info JSON",
                                paste_dialog_title="Paste machine info JSON",
                                status_initial="No machine info provided",
                                temp_prefix="machine_info_",
                                on_value_changed=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None
                            )

                        # -------------------- For this machine --------------------------
                        for_this_machine_checkbox = ui.checkbox(
                            "For this machine (run local get-initial-machine-info.sh)",
                            on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None
                        )

                # ------------------------------------------------------------------
                # Panel 2: Source repository
                # ------------------------------------------------------------------
                source_repo_card = ui.card().classes("w-full q-mt-md")
                with source_repo_card:
                    source_repo_title = ui.label("2. Source repository (optional overrides)").classes("text-subtitle1")
                    source_repo_desc = ui.label(
                        "Override the source repository URL and ref if needed. If left empty, the tool "
                        "uses values from the config or, as a fallback, the upstream of the current repo."
                    ).classes("text-body2 text-grey-7")

                    source_url_input = _style_wide(
                        ui.input("Source URL (--source-url, optional)", on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None)
                    )
                    source_ref_input = _style_wide(
                        ui.input("Source ref (--source-ref, optional)", on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None)
                    )
                    sources_target_dir_input = _style_wide(
                        ui.input("Sources target dir (--sources-target-dir, optional)", on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None)
                    )

                # ------------------------------------------------------------------
                # Panel 3.1: Repository access
                # ------------------------------------------------------------------
                clone_key_card = ui.card().classes("w-full q-mt-md")
                with clone_key_card:
                    ui.label("3.1. Repository access").classes("text-subtitle1")
                    ui.label(
                        "Specify SSH key used to access private repositories. Path is "
                        "interpreted on the machine running this UI."
                    ).classes("text-body2 text-grey-7")

                    clone_key_input = _style_wide(
                        ui.input("Clone key path (--clone-key, optional)", on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None)
                    )

                # ------------------------------------------------------------------
                # Panel 3.2.: SSH keys and machine access
                # ------------------------------------------------------------------
                ssh_card = ui.card().classes("w-full q-mt-md")
                with ssh_card:
                    ui.label("3.2. SSH keys and machine access").classes("text-subtitle1")
                    ui.label(
                        "Configure SSH keys for accessing target machines. Paths are "
                        "interpreted on the machine running this UI."
                    ).classes("text-body2 text-grey-7")

                    authorized_key_file_input = _style_wide(
                        ui.input("Authorized key file path (--authorized-key-file, optional)", on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None)
                    )
                    forward_authorized_checkbox = ui.checkbox(
                        "Forward local ~/.ssh/authorized_keys (--forward-authorized-keys)",
                        on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None
                    )
                    ui.label(
                        "Here you can specify a path where to create (or re-use, if already created) "
                        "a private key, which will be added to the authorized keys."
                    )
                    local_private_key_input = _style_wide(
                        ui.input("Local private key path (--local-private-key, optional)", on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None)
                    )
                    ui.label(
                        "Additional authorized keys to add (one per line)"
                    ).classes("text-body2 text-grey-7 q-mt-md")
                    authorized_keys_textarea = ui.textarea(
                        "Authorized keys (--authorized-keys, one per line, optional)",
                        on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None
                    ).props("rows=3").classes("w-full")

                # ------------------------------------------------------------------
                # Panel 4: Users (separate card)
                # ------------------------------------------------------------------
                users_card = ui.card().classes("w-full q-mt-md")
                with users_card:
                    ui.label("4. Users").classes("text-subtitle1")
                    users_desc_label = ui.label(
                        "Configure the main service user and optional management user."
                    ).classes("text-body2 text-grey-7")

                    user_input = _style_wide(
                        ui.input("User name (--user, optional)", on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None)
                    )
                    management_user_input = _style_wide(
                        ui.input("Management user (--management-user, optional)", on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None)
                    )

                # ------------------------------------------------------------------
                # Panel 5: Accelerator drivers (separate card)
                # ------------------------------------------------------------------
                accel_card = ui.card().classes("w-full q-mt-md")
                with accel_card:
                    ui.label("5. Accelerator drivers").classes("text-subtitle1")
                    ui.label(
                        "Configure GPU/accelerator vendor and driver installation behavior."
                    ).classes("text-body2 text-grey-7")

                    gpu_vendor_select = _style_wide(
                        ui.select(
                            {None: "Auto", **{v.vendor_name: v.vendor_name for v in GPU_VENDOR}},
                            label="GPU vendor (--gpu-vendor, optional)",
                            with_input=False,
                            on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None
                        )
                    )

                    def update_gpu_vendor_for_mode():
                        if is_simplified():
                            # In simplified mode, don't allow Auto
                            options = {v.vendor_name: v.vendor_name for v in GPU_VENDOR}
                            # If currently set to Auto (None), switch to NO
                            if gpu_vendor_select.value is None:
                                gpu_vendor_select.value = GPU_VENDOR.NO.vendor_name
                            gpu_vendor_select.options = options
                        else:
                            # In full mode, allow Auto
                            gpu_vendor_select.options = {None: "Auto", **{v.vendor_name: v.vendor_name for v in GPU_VENDOR}}

                    # Initialize the GPU vendor select based on initial mode
                    update_gpu_vendor_for_mode()

                    install_drivers_select = _style_wide(
                        ui.select(
                            {
                                "auto": "auto (detect)",
                                "yes": "yes",
                                "no": "no",
                            },
                            label="Install drivers (--install-drivers)",
                            value="auto",
                            with_input=False,
                            on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None
                        )
                    )

                # ------------------------------------------------------------------
                # Panel 6: Post-setup commands
                # ------------------------------------------------------------------
                post_commands_card = ui.card().classes("w-full q-mt-md")
                with post_commands_card:
                    ui.label("6. Post-setup commands (optional)").classes("text-subtitle1")
                    ui.label(
                        "Commands to run after setup completes (as the configured user, in the cloned "
                        "repository directory). One command per line."
                    ).classes("text-body2 text-grey-7")

                    commands_textarea = ui.textarea(
                        "Commands (one per line, optional)",
                        on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None
                    ).props("rows=4").classes("w-full")

                # ------------------------------------------------------------------
                # Panel 7: Target and profile
                # ------------------------------------------------------------------
                target_profile_card = ui.card().classes("w-full q-mt-md")
                with target_profile_card:
                    ui.label("7. Output target and profile").classes("text-subtitle1")
                    ui.label(
                        "Select the type of output to generate (cloud-init, bash script, directory with "
                        "runner, etc.) and which setup profile to apply."
                    ).classes("text-body2 text-grey-7")

                    with ui.row().classes("items-center q-gutter-md q-mt-sm"):
                        target_select = _style_wide(
                            ui.select(
                                {t.name.lower(): t.name for t in _gui_available_targets},
                                label="Export target (--target)",
                                with_input=False,
                                value="bash_script",  # default
                                on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None
                            )
                        )
                        profile_select = _style_wide(
                            ui.select(
                                {p.name.lower(): p.name for p in SetupProfile},
                                label="Profile (--profile)",
                                value="full_setup",
                                with_input=False,
                                on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None
                            )
                        )

                    dump_config_checkbox = ui.checkbox(
                        "Dump config only (--dump-config)",
                        on_change=lambda: reset_output() if not (ui_blocked_by_preset or ui_blocked_by_ssh) else None
                    )

        # ------------------------------------------------------------------
        # Panel 8: Execution and output
        # ------------------------------------------------------------------
        execution_card = ui.card().classes("w-full q-mt-md")
        with execution_card:
            ui.label("8. Execution, logs, and output").classes("text-subtitle1")
            ui.label(
                "Run the setup generation process and inspect logs, validation messages, and the "
                "generated output below. Output is shown in a card; you can copy it or download it."
            ).classes("text-body2 text-grey-7")

            status_label = ui.label("Idle")
            run_button = ui.button("Generate").classes("q-mt-sm")

            # Config display card
            config_display_card = ui.card().classes("q-mt-md w-full mb-4 border-l-4 border-blue-500")
            with config_display_card:
                ui.label("Generated/Loaded Configuration").classes("text-subtitle2 text-blue-700")
                ui.label("This is the configuration you can use to re-generate the same environment, using different setup mechanism targets.")
                config_display = ui.code("", language="json").classes("w-full")
                config_display_card.visible = False

            # simplified-mode validation warning text (no separate button)
            validation_warning_label = ui.label("Validation error").classes("text-negative q-mt-sm")
            validation_warning_label.visible = False

            # collapsible container for details (table + log)
            details_expansion = ui.expansion("Details", icon="error_outline", value=False).classes(
                "w-full q-mt-md"
            )
            with details_expansion:
                validation_card = ui.card().classes("w-full mb-4 border-l-4 border-red-500")
                with validation_card:
                    ui.label("Validation errors").classes("text-subtitle2 text-red-700")
                    validation_table = ui.table(
                        columns=[
                            {'name': 'idx', 'label': '#', 'field': 'idx', 'sortable': True,
                             'style': 'width: 10%; max-width: 20px'},
                            {'name': 'context', 'label': 'Context', 'field': 'context',
                             'style': 'width: 20%; min-width: 40px; max-width: 200px'},
                            {'name': 'message', 'label': 'Message', 'field': 'message', 'align': 'left'},
                        ],
                        rows=[],
                    ).classes("w-full").props("dense wrap-cells")

                log_output = ui.textarea("Log / validation") \
                    .props("readonly") \
                    .classes("w-full") \
                    .style("font-family: monospace; min-height: 260px")

            details_expansion.visible = False

            output_expansion = ui.expansion(
                'Generated script',
                icon='article',
                value=True,  # will be overridden dynamically based on ssh_server
            ).classes("w-full q-mt-md")

            with output_expansion:
                with ui.card().classes("w-full"):
                    ui.label("Generated script").classes("text-subtitle1")
                    output_textarea = ui.textarea("Script") \
                        .props("readonly, rows=80") \
                        .classes("w-full") \
                        .style("font-family: monospace; min-height: 260px")

                    with ui.row().classes("q-gutter-sm q-mt-sm"):
                        ui.button(
                            "Copy to clipboard",
                            on_click=lambda: ui.run_javascript(
                                f"navigator.clipboard.writeText({output_textarea.value!r})"
                            ),
                        ).props("icon=content_copy")

            output_expansion.visible = False

            # Remote execution card (initially hidden; only shown for BASH_SCRIPT + SSH target)
            remote_exec_card = ui.card().classes("w-full q-mt-md")
            with remote_exec_card:
                ui.label("Remote execution on server").classes("text-subtitle1")
                ui.label(
                    "This will SSH into the configured server as the given user "
                    "and run the generated bash script while streaming output."
                ).classes("text-body2 text-grey-7")

                remote_status_label = ui.label("Idle")

                ui.label("Execute on server from server overview:").classes("text-body2 text-grey-7 q-mt-sm")
                server_info_label = ui.label("No server configured").classes("text-body2")

                async def _run_remote() -> None:
                    nonlocal ssh_task, ui_blocked_by_ssh

                    if not ssh_server:
                        ui.notify("No SSH server configured for setup tab.", type="warning")
                        return
                    if not last_output_text:
                        ui.notify("No generated bash script to run.", type="warning")
                        return

                    # Cancel previous run (if any)
                    if ssh_task and not ssh_task.done():
                        ui.notify("Previous remote run still in progress; cancelling.", type="info")
                        ssh_task.cancel()
                        try:
                            await ssh_task
                        except Exception:
                            pass

                    # Clear previous terminal output
                    remote_log_container.clear()

                    # Create log file for SSH output
                    log_dir = Path("logs/ssh-install")
                    log_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    server_name_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', ssh_server.name or ssh_server.hostname)
                    log_file_path = log_dir / f"ssh_install_{server_name_clean}_{timestamp}.log"

                    # Update log file label
                    log_file_label.text = f"Log file: {log_file_path}"
                    log_file_label.visible = True

                    remote_status_label.text = f"Connecting to {ssh_server.hostname}..."
                    ui.notify(f"Running setup script on {ssh_server.hostname} as root", type="info")

                    ansi_re = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

                    # Open log file for writing
                    try:
                        log_file = open(log_file_path, 'w', encoding='utf-8')
                        log_file.write(f"SSH Installation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        log_file.write(f"Server: {ssh_server.name} ({ssh_server.hostname})\n")
                        log_file.write("=" * 80 + "\n\n")
                        log_file.flush()
                    except Exception as e:
                        ui.notify(f"Failed to create log file: {e}", type="warning")
                        log_file = None

                    def _log_cb(line: str) -> None:
                        # Strip ANSI escape sequences, similar to controller's usage
                        clean_line = ansi_re.sub('', line).rstrip('\n')

                        # Append as a new label in the terminal-like container
                        with remote_log_container:
                            ui.label(clean_line).classes('whitespace-pre-wrap leading-tight')

                        # Write to log file if available
                        if log_file:
                            try:
                                log_file.write(clean_line + '\n')
                                log_file.flush()
                            except Exception:
                                pass  # Silently ignore write errors

                    def _status_cb(is_running: bool) -> None:
                        remote_status_label.text = (
                            f"Running on {ssh_server.hostname}..." if is_running else "Finished / idle"
                        )

                    # Lock UI before starting task
                    set_ssh_block(True)

                    async def _runner() -> None:
                        try:
                            await run_bash_script_via_ssh(
                                server=ssh_server,
                                script_text=last_output_text,
                                log_callback=_log_cb,
                                status_callback=_status_cb,
                                force_root=False,
                            )
                        except Exception as e:
                            error_msg = f"--- Error: {e} ---"
                            with remote_log_container:
                                ui.label(error_msg).classes('whitespace-pre-wrap leading-tight')
                            if log_file:
                                try:
                                    log_file.write(f"\n{error_msg}\n")
                                    log_file.flush()
                                except Exception:
                                    pass
                        finally:
                            remote_status_label.text = "Finished / idle"
                            # Close log file
                            if log_file:
                                try:
                                    log_file.write(f"\n{'=' * 80}\n")
                                    log_file.write(f"Installation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                    log_file.close()
                                except Exception:
                                    pass
                            # Unlock UI when SSH task completes
                            set_ssh_block(False)

                    ssh_task = asyncio.create_task(_runner())


                with ui.row().classes("q-gutter-sm q-mt-sm"):
                    remote_run_button = ui.button("Run setup script on server", on_click=_run_remote).props("icon=play_arrow")

                # Label to show log file location
                log_file_label = ui.label("").classes("text-caption text-grey-7 q-mt-xs")
                log_file_label.visible = False

                ui.label("Remote console output").classes("text-body2 text-grey-7 q-mt-sm")

                remote_log_container = ui.column().classes(
                    'w-full h-128 font-mono text-sm bg-black text-white p-2 overflow-y-auto gap-0'
                )


                def update_server_info():
                    if ssh_server:
                        server_info_label.text = f"{ssh_server.name} ({ssh_server.hostname})"
                    else:
                        server_info_label.text = "No server configured"

                # Update immediately if ssh_server is already set
                update_server_info()

            remote_exec_card.visible = False

            async def _run_setup() -> None:
                nonlocal last_output_text, last_output_filename

                simple = is_simplified()

                def _path_or_none(v: str | None) -> Optional[Path]:
                    if not v:
                        return None
                    v = v.strip()
                    path = Path(v)
                    if not path.exists():
                        print(f"Error: Path does not exist: {path}", file=sys.stderr)
                    return path

                clone_key_path = _path_or_none(clone_key_input.value)
                authorized_key_file = _path_or_none(authorized_key_file_input.value)
                local_private_key = _path_or_none(local_private_key_input.value)

                sources_target_dir = (sources_target_dir_input.value or "").strip() or None
                source_url = (source_url_input.value or "").strip() or None
                source_ref = (source_ref_input.value or "").strip() or None
                user = (user_input.value or "").strip() or None
                management_user = (management_user_input.value or "").strip() or None

                raw_cmds = (commands_textarea.value or "").splitlines()
                commands = [c.strip() for c in raw_cmds if c.strip()] or None

                install_sel = install_drivers_select.value
                if install_sel == "yes":
                    install_drivers = True
                elif install_sel == "no":
                    install_drivers = False
                else:
                    install_drivers = None

                authorized_keys_raw = (authorized_keys_textarea.value or "").splitlines()

                config_path = config_json_ui.get_file_path()
                machine_info_path = machine_info_json_ui.get_file_path()
                for_this_machine = bool(for_this_machine_checkbox.value)
                dump_config = bool(dump_config_checkbox.value)
                target_name = (target_select.value or None)
                profile_name = (profile_select.value or "full_setup")

                if simple:
                    # simplified mode: sample config only, bash script target, minimal options
                    machine_info_path = None
                    for_this_machine = False
                    source_url = None
                    source_ref = None
                    clone_key_path = None
                    authorized_key_file = None
                    management_user = None
                    commands = None
                    dump_config = False
                    profile_name = "full_setup"
                    target_name = "bash_script"

                args = dict(
                    config_path=config_path,
                    machine_info_file=machine_info_path,
                    for_this_machine=for_this_machine,
                    dump_config=dump_config,
                    target_name=target_name,
                    profile_name=profile_name,
                    source_url=source_url,
                    source_ref=source_ref,
                    sources_target_dir=sources_target_dir,
                    out_path=None,  # no filesystem output; show in card instead
                    commands=commands,
                    clone_key_path=clone_key_path,
                    authorized_key_file=authorized_key_file,
                    authorized_keys=[k.strip() for k in authorized_keys_raw if k.strip()] or [],
                    forward_authorized_keys=bool(forward_authorized_checkbox.value),
                    local_private_key=local_private_key,
                    user=user,
                    management_user=management_user,
                    gpu_vendor_name=(gpu_vendor_select.value or None),
                    install_drivers=install_drivers,
                )

                status_label.text = "Running..."
                log_output.value = ""
                validation_table.rows = []
                validation_warning_label.visible = False
                details_expansion.visible = False
                details_expansion.value = False
                output_expansion.visible = False
                config_display_card.visible = False
                last_output_text = ""
                last_output_filename = "setup_output.txt"

                try:
                    res = run_setup_from_gui(**args)

                    combined_log: List[str] = []
                    combined_log.extend(res.logs)
                    combined_log.append("\n--- Validation ---\n")
                    combined_log.extend(res.validation.logs)
                    if res.validation.errors:
                        combined_log.append(
                            f"\nSummary: {len(res.validation.errors)} validation error(s) recorded."
                        )
                    log_output.value = "\n".join(combined_log)

                    rows = [
                        {
                            'idx': i + 1,
                            'context': err.context,
                            'message': err.message,
                        }
                        for i, err in enumerate(res.validation.errors)
                    ]
                    validation_table.rows = rows

                    if rows:
                        details_expansion.visible = True
                        if simple:
                            # Simplified: show red text, keep details collapsed until user expands.
                            validation_warning_label.visible = True
                            details_expansion.value = False
                        else:
                            validation_warning_label.visible = False
                            details_expansion.value = True
                    else:
                        validation_warning_label.visible = False
                        details_expansion.visible = False
                        details_expansion.value = False

                    last_output_text = res.output_text or ""
                    if res.target is None and bool(dump_config_checkbox.value):
                        last_output_filename = "config.json"
                    elif res.target is not None:
                        if res.target.name == "CLOUD_INIT":
                            last_output_filename = "cloud_init.yaml"
                        elif res.target.name == "BASH_SCRIPT":
                            last_output_filename = "setup.sh"
                        elif res.target.name == "DIR_WITH_RUNNER":
                            last_output_filename = "dir_with_runner.txt"
                        elif res.target.name == "SKY_LAUNCH":
                            last_output_filename = "skypilot_launch.yaml"
                        else:
                            last_output_filename = "setup_output.txt"
                    else:
                        last_output_filename = "setup_output.txt"

                    if last_output_text:
                        output_textarea.value = last_output_text
                        output_expansion.visible = True

                        # Collapse if SSH runner is available, expand otherwise
                        if ssh_server is not None:
                            output_expansion.value = False   # collapsed
                        else:
                            output_expansion.value = True    # expanded
                    else:
                        output_expansion.visible = False

                    # Show/hide remote run card:
                    if (
                        last_output_text
                        and res.target == ExportTarget.BASH_SCRIPT
                        and ssh_server is not None
                    ):
                        update_server_info()
                        remote_exec_card.visible = True
                    else:
                        remote_exec_card.visible = False

                    if res.cfg:
                        config_json = json.dumps(res.cfg.to_dict(), indent=2)
                        config_display.set_content(config_json)
                        config_display_card.visible = True

                    if res.target is None and bool(dump_config_checkbox.value):
                        status_label.text = "Finished: config generated (not written to disk)."
                    elif res.target is not None:
                        status_label.text = (
                            f"Finished: target={res.target.name}, "
                            f"profile={res.profile.name if res.profile else '?'}"
                        )
                    else:
                        status_label.text = "Finished."
                except Exception as e:
                    print("="*80, file=sys.stderr)
                    print("= Details for non-fatal validation error", file=sys.stderr)
                    print("="*80, file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    print("="*80, file=sys.stderr)
                    log_output.value = (log_output.value or "") + f"\nException: {e}"
                    status_label.text = "Error"

            run_button.on_click(_run_setup)

        # mode-dependent visibility and texts
        def update_mode() -> None:
            simple = is_simplified()

            # base descriptions
            if simple:
                main_desc_label.text = (
                    "Generate a bash setup script from a selected sample configuration. "
                    "This is a simplified wrapper around the same core logic as the command-line tool."
                )
                base_title_label.text = "1. Sample configuration"
                base_desc_label.text = (
                    "Select a sample configuration JSON as the basis for generating a bash setup script."
                )
                config_title_label.text = "Sample configuration"
                config_help_label.text = "Select one of the predefined sample configurations."
                users_desc_label.text = "Configure the main service user name used in the generated script."
            else:
                main_desc_label.text = (
                    "Generate setup scripts or configuration for target machines from either a full config "
                    "JSON or machine info JSON. This is a GUI wrapper around the same core logic as the "
                    "command-line tool."
                )
                base_title_label.text = "1. Base configuration / machine info"
                base_desc_label.text = (
                    "Provide either: (A) a full config JSON, or (B) machine info JSON captured on the "
                    "target machine. If 'For this machine' is enabled, local machine info is collected "
                    "automatically using the reference script."
                )
                config_title_label.text = "Config JSON (optional)"
                config_help_label.text = (
                    "If provided, this config will be used as the starting point and updated according "
                    "to other options. If omitted, the tool can derive a config from machine info."
                )
                users_desc_label.text = "Configure the main service user and optional management user."

            # base config / machine info visibility
            and_or_row.visible = not simple
            machine_info_section.visible = not simple
            for_this_machine_checkbox.visible = not simple

            # source repo and clone key card
            source_url_input.visible = not simple
            source_ref_input.visible = not simple
            source_repo_title.text = "2. Source directory" if simple else "2. Source repository (optional overrides)"
            source_repo_desc.text = (
                "Specify the target directory name for the source code."
            ) if simple else (
                "Override the source repository URL and ref if needed. If left empty, the tool "
                "uses values from the config or, as a fallback, the upstream of the current repo."
            )
            clone_key_card.visible = not simple

            # SSH advanced options
            authorized_key_file_input.visible = not simple

            # management user and driver installation
            management_user_input.visible = not simple
            install_drivers_select.visible = not simple

            # post-setup commands
            post_commands_card.visible = not simple

            # target / profile / dump-config
            target_profile_card.visible = not simple
            profile_select.visible = not simple
            dump_config_checkbox.visible = not simple

            # in simplified mode always use bash script target
            if simple:
                target_select.value = "bash_script"

            # validation + log details (initially hidden until a run)
            validation_warning_label.visible = False
            details_expansion.visible = False
            details_expansion.value = False

            # JSON source UIs: hide upload/paste in simplified mode
            config_json_ui.set_simplified(simple)
            # machine_info_json_ui is entirely hidden in simplified mode, keep full capabilities when visible
            machine_info_json_ui.set_simplified(False)

        mode_toggle.on_value_change(lambda e: update_mode())
        update_mode()


        # helper that can be used from outside
        def _set_simplified(flag: bool) -> None:
            mode_toggle.value = 'simplified' if flag else 'full'
            update_mode()
            # also keep JSON widgets in consistent state
            config_json_ui.set_simplified(flag)

        def _preset_simplified(username: str, gpu_vendor_name: str, config_sample: Optional[str], private_key_file: Optional[str], sources_target_dir: Optional[str]) -> None:
            nonlocal ui_blocked_by_preset

            # Temporarily disable blocking for reset
            temp_blocked = ui_blocked_by_preset
            ui_blocked_by_preset = False
            reset_output()
            ui_blocked_by_preset = temp_blocked

            # 1) switch the tab itself into simplified mode
            _set_simplified(True)

            # 2) select the sample config (if given)
            if config_sample:
                config_json_ui.select_sample(config_sample)

            # 3) set user name
            user_input.value = username

            # 4) set GPU vendor
            # in simplified mode Auto is removed; use explicit vendor name
            gpu_vendor_select.value = gpu_vendor_name

            # 5) set private key file
            if private_key_file:
                local_private_key_input.value = private_key_file

            # 6) set sources target directory
            if sources_target_dir:
                sources_target_dir_input.value = sources_target_dir

            preset_info_card.visible = True
            
            # Lock config inputs when preset is applied, but keep sample selector enabled
            ui_blocked_by_preset = True
            set_config_inputs_enabled(False, keep_sample_select_enabled=True)
            # Keep run buttons enabled

        def _set_ssh_server(server: Optional[Server]) -> None:
            nonlocal ssh_server
            ssh_server = server

        return SetupTabHandle(
            preset_simplified=_preset_simplified,
            set_ssh_server=_set_ssh_server,
        )
