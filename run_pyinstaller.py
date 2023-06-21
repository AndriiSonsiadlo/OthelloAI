from pathlib import Path

import PyInstaller.__main__


def compile_app():
    app_folder = Path(__file__).parent
    main_file = app_folder / "src/main.py"
    file = app_folder / "Othello.exe"
    file.unlink(missing_ok=True)
    PyInstaller.__main__.run(
        [
            str(main_file.resolve()),
            "--name=Othello",
            "--clean",
            "--onefile"
        ]
    )


def main():
    compile_app()


if __name__ == '__main__':
    main()
