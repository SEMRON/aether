#!/usr/bin/env python3
import os
os.environ['distqat_skip_model_load'] = '1'
from src.distqat.gui.layout import init_ui
from src.distqat.gui.icons import ICON_SVG
from src.distqat.gui.colors import PRIMARY_COLOR, SECONDARY_COLOR, BACKGROUND_COLOR, POSITIVE_COLOR, NEGATIVE_COLOR, INFO_COLOR, DARK_COLOR, DARK_PAGE_COLOR, ACCENT_COLOR, WARNING_COLOR
from nicegui import ui
import argparse
from run_script_utils import is_wandb_logged_in

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()


    if not args.no_wandb and not is_wandb_logged_in():
        raise RuntimeError("Wandb is not logged in, please login to wandb using the wandb login command or set the wandb_project to None through the config file or command line argument")

    ui.colors(
        primary=PRIMARY_COLOR,
        secondary=SECONDARY_COLOR,
        accent=ACCENT_COLOR,
        dark=DARK_COLOR,
        dark_page=DARK_PAGE_COLOR,
        positive=POSITIVE_COLOR,
        negative=NEGATIVE_COLOR,
        info=INFO_COLOR,
        warning=WARNING_COLOR,
        custom_colors={
            'background': BACKGROUND_COLOR
        }
    )

    init_ui(args.no_wandb)
    ui.run(
        title="DistQAT Dashboard",
        port=args.port,
        host=args.host,
        favicon=ICON_SVG,
        dark=True
    )

if __name__ in {"__main__", "__mp_main__"}:
    main()
