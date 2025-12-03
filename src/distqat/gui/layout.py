from nicegui import ui, app
import asyncio
from pathlib import Path
import glob
from typing import List
from datetime import datetime

from .config import Server, JobConfig
from .controller import Controller
from .ssh_runner import SSHSession

from .machine_setup import create_setup_tab

class DistributedGui:
    def __init__(self, no_wandb: bool = False):
        self.controller = Controller(no_wandb)
        self.controller.log_callback = self.on_log
        self.controller.status_callback = self.on_status
        self.controller.server_status_callback = self.on_server_status
        self.controller.error_callback = self.on_error
        self.controller.error_reset_callback = self.on_error_reset


        self.log_elements = {}  # server_name -> ui.log
        self.worker_start_btns = {} # server_name -> ui.button
        self.worker_status_indicators = {} # server_name -> ui.icon
        self.detected_errors = [] # List of error dicts

        self.setup_ui()
        self.refresh_server_list()  # Initial refresh to populate from loaded state

    def on_error_reset(self, source: str):
        # Remove all errors from this source
        original_count = len(self.detected_errors)
        self.detected_errors = [e for e in self.detected_errors if e['source'] != source]
        if len(self.detected_errors) != original_count:
             self.refresh_error_list()

    def on_error(self, event):
        # event is dict: {source, message, context, timestamp}

        # Check if we already have an error for this source
        # Only keep one error per source to avoid flooding
        if any(e['source'] == event['source'] for e in self.detected_errors):
            return

        self.detected_errors.insert(0, event) # Add to top
        # Keep list size manageable
        if len(self.detected_errors) > 100:
            self.detected_errors.pop()

        # Update UI if it exists
        if hasattr(self, 'error_list'):
            self.refresh_error_list()

        # Notify
        if hasattr(self, 'header'):
            with self.header:
                ui.notify(f"Error detected in {event['source']}", type='negative')
        else:
            print(f"[ERROR] Error detected in {event['source']}")

    def refresh_error_list(self):
        if not hasattr(self, 'error_list'):
            return

        if hasattr(self, 'header'):
            with self.header:
                self.error_list.clear()
                with self.error_list:
                    for err in self.detected_errors:
                        with ui.expansion(f"{err['timestamp']} - {err['source']}", icon='error').classes('w-full bg-red-50'):
                            with ui.column().classes('p-2'):
                                ui.label(f"Message: {err['message']}").classes('font-bold text-red-800')
                                ui.label("Context:").classes('font-bold mt-2')
                                ui.code(err['context']).classes('w-full')

    def on_server_status(self, server_name: str, is_running: bool):
        if server_name in self.worker_status_indicators:
            icon = self.worker_status_indicators[server_name]
            if is_running:
                icon.props('name=check_circle color=green')
                msg = f"{server_name} is running"
                typ = 'positive'
            else:
                icon.props('name=error color=red')
                msg = f"{server_name} stopped"
                typ = 'warning'

            # Ensure context is set for background notifications
            if hasattr(self, 'header'):
                with self.header:
                    ui.notify(msg, type=typ)
            else:
                print(f"[{typ.upper()}] {msg}")

    def on_log(self, server_name: str, line: str):
        # print(f"Log from {server_name}: {line}")
        if server_name in self.log_elements:
            if hasattr(self, 'header'):
                with self.header:
                    with self.log_elements[server_name]:
                        ui.label(line).classes('whitespace-pre-wrap leading-tight')

    def on_status(self, message: str):
        # Ensure context is set for background notifications
        if hasattr(self, 'header'):
            with self.header:
                ui.notify(message)
        else:
            print(f"[STATUS] {message}")

    def update_log_views(self):
        # Add new tabs for new sessions
        if not hasattr(self, 'log_tabs') or not hasattr(self, 'log_panels'):
            return

        for name in list(self.controller.sessions.keys()):
            if name not in self.log_elements:
                with self.log_tabs:
                    tab = ui.tab(name)
                with self.log_panels:
                    with ui.tab_panel(tab):
                        with ui.row().classes('w-full justify-between items-center p-2 bg-gray-50'):
                            ui.label(f"Session: {name}").classes('font-bold')
                            async def stop_node(n=name):
                                await self.controller.stop_task(n)
                            ui.button('Stop Node', on_click=stop_node, color='red-4').classes('text-white px-4')

                            self.log_elements[name] = ui.column().classes('w-full h-full font-mono text-sm bg-black text-white p-2 overflow-y-auto gap-0')

                if self.log_tabs.value is None:
                    self.log_tabs.value = name



    def refresh_server_list(self):
        if hasattr(self, 'server_select'):
            self.server_select.options = [s.display_name for s in self.controller.state.servers]
            self.server_select.update()

        if hasattr(self, 'workers_container'):
            self.workers_container.clear()
            self.worker_start_btns = {}
            with self.workers_container:
                for s in self.controller.state.servers:
                    with ui.card().classes('w-full p-2'):
                            with ui.row().classes('w-full items-center justify-between no-wrap'):
                                with ui.row().classes('items-center gap-2 w-1/4'):
                                    self.worker_status_indicators[s.display_name] = ui.icon('circle', color='grey').tooltip('Status')
                                    ui.label(s.display_name).classes('font-bold')

                                # Inputs
                            with ui.row().classes('gap-2'):
                                def update_server_config(server=s, field=None, value=None):
                                    if field and value is not None:
                                        setattr(server, field, value)
                                        self.controller.save_state()

                                n_servers = ui.number('Servers', value=s.num_servers, min=1, format='%.0f',
                                                    on_change=lambda e, s=s: update_server_config(s, 'num_servers', int(e.value))).classes('w-20').tooltip('Number of servers to be started')



                                batch_sz = ui.number('Batch', value=s.batch_size, min=1, format='%.0f',
                                                   on_change=lambda e, s=s: update_server_config(s, 'batch_size', int(e.value))).classes('w-20').tooltip('Batch Size')
                                
                                inner_steps = ui.number('Inner Steps', value=s.inner_steps, min=1, format='%.0f',
                                                   on_change=lambda e, s=s: update_server_config(s, 'inner_steps', int(e.value))).classes('w-20').tooltip('Inner Steps')


                            with ui.row().classes('gap-2'):
                                async def start(name=s.display_name):
                                    server = self.controller.get_server(name)
                                    try:
                                        await self.controller.start_worker_nodes([name], server.num_servers, server.device, server.batch_size, server.inner_steps, server.grpc_announce_port, server.mapped_host_port)
                                        self.update_log_views()
                                        ui.notify(f"Started worker on {name}")
                                    except Exception as e:
                                        ui.notify(str(e), type='negative')

                                btn = ui.button('Start', on_click=start, color='primary').classes('small')
                                # Initial state check
                                if not self.controller.initial_peers:
                                    btn.disable()
                                else:
                                    btn.enable()

                                self.worker_start_btns[s.display_name] = btn

                                async def stop(name=s.display_name):
                                     sessions = [sid for sid, sess in self.controller.sessions.items() if sess.server.display_name == name]
                                     if not sessions:
                                         ui.notify(f"No active sessions found for {name}", type='warning')
                                         return
                                     for sid in sessions:
                                         await self.controller.stop_task(sid)
                                     ui.notify(f"Stopped workers on {name}")

                                ui.button('Stop', on_click=stop, color='red-4').classes('small')

        if hasattr(self, 'server_table'):
            rows = []
            for s in self.controller.state.servers:
                row = s.model_dump()
                row['name'] = s.display_name
                rows.append(row)
            self.server_table.rows = rows
            self.server_table.update()

    async def test_connection(self, server: Server):
        ui.notify(f"Testing connection to {server.hostname}...")
        session = SSHSession(server)
        success = await session.test_connection()
        if success:
            ui.notify(f"Connection to {server.hostname} successful!", type='positive')
        else:
            ui.notify(f"Connection to {server.hostname} failed.", type='negative')


    async def show_file_content(self, row_data):
        # Cancel any existing polling task
        if hasattr(self, '_poll_task') and self._poll_task:
             self._poll_task.cancel()

        if not row_data or 'url' not in row_data:
             return

        name = row_data['name']
        url = row_data['url']
        self.file_title.text = f"File: {name}"
        self.file_content.clear()
        with self.file_content:
            ui.label("Loading...").classes('whitespace-pre-wrap leading-tight')
        self.file_dialog.open()

        async def _poll_content():
            try:
                while self.file_dialog.value:
                    try:
                        content = await self.controller.get_file_content(url)

                        # Check if dialog is still open and we are still the active task
                        if not self.file_dialog.value:
                            print("Dialog closed")
                            break

                        if not hasattr(self, '_last_file_content') or self._last_file_content != content:
                            self.file_content.clear()
                            with self.file_content:
                                ui.label(content).classes('whitespace-pre-wrap leading-tight')
                            self._last_file_content = content

                            # Scan for errors
                            file_source_name = f"File: {name}"
                            self.controller.scan_text_for_errors(content, file_source_name)

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        self.file_content.push(f"Error loading file: {e}")
                        break

                    await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                pass

        # Start polling task
        self._poll_task = asyncio.create_task(_poll_content())

        # Handle dialog close to stop polling
        def on_close():
            print("On close")
            if hasattr(self, '_poll_task') and self._poll_task:
                self._poll_task.cancel()
                self._poll_task = None
            if hasattr(self, '_last_file_content'):
                del self._last_file_content

        # Clear existing listeners to avoid duplicate close handlers
        self.file_dialog.on('close', on_close)


    def on_main_tab_change(self, e):
        if e.value == 'Monitor':
            if hasattr(self, 'log_tabs') and self.log_tabs.value is None and self.log_elements:
                self.log_tabs.value = list(self.log_elements.keys())[0]

    def setup_ui(self):
        with ui.header().classes('bg-primary text-white') as header:
            self.header = header
            ui.label('DistQAT Control Center').classes('text-h5')

        with ui.tabs().on_value_change(self.on_main_tab_change).classes('w-full') as tabs:
            setup_tab = ui.tab("Machine Setup Helper")
            server_tab = ui.tab('Servers')
            orchestrate_tab = ui.tab('Orchestrate')
            monitor_tab = ui.tab('Monitor')

        with ui.tab_panels(tabs, value=setup_tab).classes('w-full'):
            setup_tab_handle = create_setup_tab(setup_tab)

        with ui.tab_panels(tabs, value=server_tab).classes('w-full'):

            # --- Server Management ---
            with ui.tab_panel(server_tab):
                with ui.row().classes('w-full gap-4'):
                    with ui.card().classes('flex-1'):
                        ui.label('Add/Edit Server').classes('text-h6')

                        self.hostname_input = ui.input('Hostname / IP').classes('w-2/3')
                        self.ssh_port_input = ui.number('SSH Port', value=22, format='%.0f').classes('w-2/3')
                        self.username_input = ui.input('Username', value='root').classes('w-2/3')
                        self.key_path_input = ui.input('SSH Key Path (Optional)').classes('w-2/3').tooltip('Path to the SSH key file to be used for authentication to the server')
                        self.remote_dir_input = ui.input('Remote Root Dir', value='~/distqat').classes('w-2/3').tooltip('Remote directory where distqat is or will be installed')
                        self.name_input = ui.input('Display Name (Optional)').classes('w-2/3')

                        ui.separator().classes('my-2 w-2/3')
                        ui.label('Process Config').classes('text-subtitle2')

                        with ui.row().classes('w-2/3 gap-2'):
                            self.device_input = ui.select(['cpu', 'cuda', 'rocm'], value='cpu', label='Device').classes('w-1/3')
                            with ui.expansion("Port mapping", icon="lan").classes("w-full flex-1"):
                                self.grpc_port_input = ui.number('GRPC Port', placeholder='Auto', format='%.0f').classes('w-full').tooltip('Base GRPC Announce Port')
                                self.monitor_port_input = ui.number('Monitor Port', placeholder='Default', format='%.0f').classes('w-full').tooltip('Mapped Monitor Port')
                                self.host_port_input = ui.number('Host Port', placeholder='Default', format='%.0f').classes('w-full').tooltip('Mapped Server Host Port (for DHT peer discovery)')

                        self.editing_server_name = None

                        def clear_form():
                            self.hostname_input.value = ''
                            self.ssh_port_input.value = 22
                            self.username_input.value = 'root'
                            self.key_path_input.value = ''
                            self.remote_dir_input.value = '~/distqat'
                            self.name_input.value = ''
                            self.device_input.value = 'cpu'
                            self.grpc_port_input.value = None
                            self.monitor_port_input.value = None
                            self.host_port_input.value = None
                            self.editing_server_name = None
                            self.add_btn.text = 'Add Server'

                        def add_server():
                            s = Server(
                                hostname=self.hostname_input.value.strip(),
                                ssh_port=int(self.ssh_port_input.value),
                                username=self.username_input.value.strip(),
                                key_path=self.key_path_input.value.strip() if self.key_path_input.value else None,
                                remote_root_dir=self.remote_dir_input.value.strip(),
                                name=self.name_input.value.strip() if self.name_input.value else None,
                                device=self.device_input.value,
                                grpc_announce_port=int(self.grpc_port_input.value) if self.grpc_port_input.value else None,
                                mapped_monitor_port=int(self.monitor_port_input.value) if self.monitor_port_input.value else None,
                                mapped_host_port=int(self.host_port_input.value) if self.host_port_input.value else None
                            )

                            if self.editing_server_name:
                                self.controller.remove_server(self.editing_server_name)
                                ui.notify(f'Server updated')
                            else:
                                ui.notify('Server added')

                            self.controller.add_server(s)
                            self.refresh_server_list()
                            clear_form()

                        with ui.row().classes('w-full'):
                            self.add_btn = ui.button('Add Server', on_click=add_server).classes('flex-1')
                            ui.button('Clear', on_click=clear_form, color='grey').classes('w-1/3')

                    with ui.card().classes('flex-2'):
                        ui.label('Configured Servers').classes('text-h6')

                        columns = [
                            {'name': 'name', 'label': 'Name', 'field': 'name', 'sortable': True},
                            {'name': 'hostname', 'label': 'Host', 'field': 'hostname', 'sortable': True},
                            {'name': 'username', 'label': 'User', 'field': 'username'},
                            {'name': 'ssh_port', 'label': 'SSH Port', 'field': 'ssh_port'},
                            {'name': 'device', 'label': 'Device', 'field': 'device'},
                        ]
                        self.server_table = ui.table(columns=columns, rows=[], row_key='name', selection='single').classes('w-full')


                        def delete_server():
                            if self.server_table.selected:
                                selected_row = self.server_table.selected[0]
                                name_to_delete = selected_row['name']
                                self.controller.remove_server(name_to_delete)
                                self.refresh_server_list()
                                self.server_table.selected = [] # Clear selection
                                ui.notify(f'Server {name_to_delete} deleted')
                            else:
                                ui.notify('Please select a server to delete', type='warning')

                        def edit_server():
                            if self.server_table.selected:
                                selected_row = self.server_table.selected[0]
                                name = selected_row['name']
                                server = self.controller.get_server(name)
                                if server:
                                    self.hostname_input.value = server.hostname
                                    self.ssh_port_input.value = server.ssh_port
                                    self.username_input.value = server.username
                                    self.key_path_input.value = server.key_path or ''
                                    self.remote_dir_input.value = server.remote_root_dir
                                    self.name_input.value = server.name or ''
                                    self.device_input.value = server.device
                                    self.grpc_port_input.value = server.grpc_announce_port
                                    self.monitor_port_input.value = server.mapped_monitor_port
                                    self.host_port_input.value = server.mapped_host_port

                                    self.editing_server_name = name
                                    self.add_btn.text = 'Update Server'
                                    ui.notify(f'Editing {name}')
                            else:
                                ui.notify('Please select a server to edit', type='warning')

                        def create_server_setup_script():
                            if self.server_table.selected:
                                selected_row = self.server_table.selected[0]
                                name = selected_row['name']
                                server = self.controller.get_server(name)
                                if server:
                                    from .machine_setup import GPU_VENDOR
                                    if server.device == 'cuda':
                                        gpu_vendor = GPU_VENDOR.NVIDIA.vendor_name
                                    elif server.device == 'rocm':
                                        gpu_vendor = GPU_VENDOR.AMD.vendor_name
                                    else:
                                        gpu_vendor = GPU_VENDOR.NO.vendor_name

                                    setup_tab_handle.preset_simplified(
                                        username=server.username,
                                        gpu_vendor_name=gpu_vendor,
                                        config_sample="ubuntu-default",
                                        private_key_file=server.key_path,
                                        sources_target_dir=server.remote_root_dir,
                                    )
                                    selected_row = self.server_table.selected[0]
                                    name = selected_row['name']
                                    server = self.controller.get_server(name)
                                    setup_tab_handle.set_ssh_server(server)
                                    tabs.value = setup_tab
                                    ui.notify(f'Setup script configured for {name}')
                                    tabs.value = setup_tab
                            else:
                                ui.notify('Please select a server to create setup script', type='warning')

                        with ui.row().classes('w-full gap-2'):
                            ui.button('Edit Selected Server', on_click=edit_server, color='primary').classes('flex-1')
                            ui.button('Delete Selected Server', on_click=delete_server, color='red-4').classes('flex-1')

                        async def test_ssh_conn():
                            if self.server_table.selected:
                                selected_row = self.server_table.selected[0]
                                server = self.controller.get_server(selected_row['name'])
                                if server:
                                    await self.test_connection(server)
                            else:
                                ui.notify('Please select a server', type='warning')


                        with ui.row().classes('w-full gap-2'):
                            ui.button('Test SSH Connection', on_click=test_ssh_conn, color='primary').classes('flex-1')
                            ui.button('Create Server Setup', on_click=create_server_setup_script, color='primary').classes('flex-1')

            # --- Orchestration ---
            with ui.tab_panel(orchestrate_tab):
                with ui.row().classes('w-full'):
                    with ui.column().classes('w-1/3'):
                        ui.label('Configuration').classes('text-h6')

                        # Scan for configs
                        config_files = glob.glob("configs/*.yaml")
                        config_select = ui.select(config_files, label='Config File', value=self.controller.state.last_job_config.config_path).classes('w-full')

                        wandb_key = ui.input('WandB API Key', password=True).classes('w-full')
                        if self.controller.state.last_job_config.wandb_api_key:
                            wandb_key.value = self.controller.state.last_job_config.wandb_api_key

                        ui.separator().classes('my-4')

                        ui.label('Head Node (Client/Monitor)').classes('text-h6')
                        with ui.row().classes('w-full gap-2'):
                            self.server_select = ui.select([], label='Select Head Node').classes('flex-1')

                        ui.separator().classes('my-2')
                        ui.label('Worker Nodes').classes('text-h6')
                        self.workers_container = ui.column().classes('w-full gap-2')


                    with ui.column().classes('w-2/3'):
                        ui.label('Controls').classes('text-h6')

                        with ui.row():
                            async def start_head():
                                cfg = JobConfig(
                                    config_path=config_select.value,
                                    wandb_api_key=wandb_key.value
                                )
                                await self.controller.start_head_node(self.server_select.value, cfg)
                                self.update_log_views()
                                self.peers_display.content = "Waiting for peers..."

                            ui.button('1. Start Head Node', on_click=start_head, color='primary')

                            async def start_all_workers():
                                for s in self.controller.state.servers:
                                    try:
                                        await self.controller.start_worker_nodes([s.display_name], s.num_servers, s.device, s.batch_size, s.inner_steps, s.grpc_announce_port, s.mapped_host_port)
                                        ui.notify(f"Started worker on {s.display_name}")
                                    except Exception as e:
                                        ui.notify(f"Failed to start {s.display_name}: {e}", type='negative')
                                self.update_log_views()

                            self.start_all_btn = ui.button('2. Start All Workers', on_click=start_all_workers, color='primary')
                            self.start_all_btn.disable()

                            async def stop_all():
                                await self.controller.stop_all()
                                self.peers_display.content = "Stopped"

                            self.stop_all_btn = ui.button('STOP ALL', on_click=stop_all, color='red-4')
                            self.stop_all_btn.disable()

                        async def reset_all():
                             await self.controller.reset_all_processes()
                             self.peers_display.content = "Reset"

                        ui.button('Reset all processes', on_click=reset_all, color='red-10').classes('text-white w-1/3 mt-2')

                        ui.label('Initial Peers:').classes('text-lg mt-4 font-bold')
                        with ui.row().classes('w-full items-center gap-2 bg-gray-100 p-2 rounded mb-4'):
                            self.peers_display = ui.code('Not started').classes('flex-1')


                        # Update peers label periodically
                        def update_peers():
                            if self.controller.initial_peers:
                                self.peers_display.content = str(self.controller.initial_peers)
                                # Enable all worker start buttons
                                self.start_all_btn.enable()
                                for btn in self.worker_start_btns.values():
                                    btn.enable()
                            else:
                                self.start_all_btn.disable()
                                for btn in self.worker_start_btns.values():
                                    btn.disable()

                            if self.controller.head_server:
                                self.stop_all_btn.enable()
                            else:
                                self.stop_all_btn.disable()
                        ui.timer(1.0, update_peers)


            # --- Monitoring ---
            with ui.tab_panel(monitor_tab):
                # Error Watcher
                with ui.card().classes('w-full mb-4 border-l-4 border-red-500'):
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label('Error Watcher').classes('text-h6 text-red-700')
                        ui.button('Clear', on_click=lambda: (self.detected_errors.clear(), self.refresh_error_list()), color='grey').classes('small')

                    self.error_list = ui.column().classes('w-full gap-1 overflow-y-auto max-h-80')
                    self.refresh_error_list()

                ui.label('Live Logs').classes('text-h6')

                with ui.row().classes('w-full'):
                    # Create a log area for each potential server
                    # For simplicity, we recreate tabs or just show all
                    # Let's use a tabs approach for logs

                    self.log_tabs = ui.tabs().classes('w-full')
                    self.log_panels = ui.tab_panels(self.log_tabs).classes('w-full h-96 border')

                self.update_log_views()

                # --- WandB Metrics & Files ---
                ui.separator().classes('my-4')

                with ui.row().classes('w-full gap-4'):
                    # Chart
                    with ui.card().classes('flex-1 h-96'):
                        ui.label('Training Metrics').classes('text-h6')
                        # Basic echart configuration
                        self.chart = ui.echart({
                            'title': {'text': 'Distributed Training Loss'},
                            'tooltip': {'trigger': 'axis'},
                            'legend': {'data': ['loss/distributed']},
                            'xAxis': {'type': 'category', 'data': []},
                            'yAxis': {'type': 'value'},
                            'series': [{'name': 'loss/distributed', 'type': 'line', 'data': []}]
                        }).classes('w-full h-full')

                    # Files Table
                    with ui.card().classes('flex-1 h-96'):
                        ui.label('Log Files').classes('text-h6')
                        columns = [
                            {'name': 'name', 'label': 'File Name', 'field': 'name'},
                            {'name': 'size', 'label': 'Size (bytes)', 'field': 'size'},
                            {'name': 'updated_at', 'label': 'Updated', 'field': 'updated_at'},
                        ]
                        self.files_table = ui.table(columns=columns, rows=[], row_key='name').classes('w-full h-full')

                        # File content dialog
                        self.file_dialog = ui.dialog()
                        with self.file_dialog, ui.card().classes('w-3/4 max-w-6xl h-3/4'):
                            with ui.row().classes('w-full justify-between items-center'):
                                self.file_title = ui.label('File Content').classes('text-h6')
                                ui.button('Close', on_click=self.file_dialog.close, color='red-4')
                            self.file_content = ui.column().classes('w-full h-full font-mono text-sm bg-black text-white p-2 overflow-y-auto gap-0')

                        self.files_table.add_slot('body-cell-name', r'''
                            <q-td :props="props">
                                <a href="#" @click.prevent="$parent.$emit('open_file', props.row)" class="text-blue-500 hover:underline">
                                    {{ props.row.name }}
                                </a>
                            </q-td>
                        ''')

                        self.files_table.on('open_file', lambda e: asyncio.create_task(self.show_file_content(e.args)))

                async def update_wandb_data():
                    if not self.controller.wandb_run_path:
                        return

                    # Update Chart
                    metrics = await self.controller.get_wandb_metrics(keys=["loss/distributed", "monitor/step"])
                    if metrics and "loss/distributed" in metrics:
                        steps = metrics.get("monitor/step", list(range(len(metrics["loss/distributed"]))))
                        self.chart.options['xAxis']['data'] = steps
                        self.chart.options['series'][0]['data'] = metrics["loss/distributed"]
                        self.chart.update()

                    # Update Files
                    files = await self.controller.get_wandb_files()
                    if files:
                        self.files_table.rows = files
                        self.files_table.update()

                        # Passive scan
                        await self.controller.check_wandb_files_for_errors(files)

                ui.timer(5.0, update_wandb_data) # Poll every 5 seconds


                # WandB Embed
                ui.separator().classes('my-4')
                ui.label('WandB Dashboard').classes('text-h6')

                self.wandb_link = ui.link('Open WandB (Waiting for run...)', 'https://wandb.ai').classes('text-blue-500 text-lg')

                def update_wandb_link():
                    if self.controller.wandb_run_url:
                        self.wandb_link.text = "Open WandB Run"
                        self.wandb_link.props['href'] = self.controller.wandb_run_url
                    else:
                        self.wandb_link.text = "Open WandB (Waiting for run...)"
                        self.wandb_link.props['href'] = "https://wandb.ai"

                ui.timer(2.0, update_wandb_link)



def init_ui(no_wandb: bool = False):
    DistributedGui(no_wandb)
