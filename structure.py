import os
import sys


def display_app_structure(startpath, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', 'venv', 'env']

    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
        try:
            print(f'{indent}{os.path.basename(root)}/')
        except UnicodeEncodeError:
            print(f'{indent}[Undecodable directory name]/')
        sub_indent = '│   ' * level + '├── '
        for file in files:
            try:
                print(f'{sub_indent}{file}')
            except UnicodeEncodeError:
                print(f'{sub_indent}[Undecodable filename]')


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("App Structure:")

    # Set console to UTF-8 mode
    if sys.platform.startswith('win'):
        import subprocess
        subprocess.run(["chcp", "65001"], shell=True)

    sys.stdout.reconfigure(encoding='utf-8')
    display_app_structure(current_dir)
