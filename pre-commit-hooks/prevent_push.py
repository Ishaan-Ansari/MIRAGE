import subprocess
import sys

# List of protected branch
PROTECTED_BRANCHES = ['dev', 'staging', 'prod']

def get_current_branch():
    """Get the current branch name"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error determining the current branch name: {e.stderr}")
        sys.exit(1)

def main():
    current_branch = get_current_branch()

    if current_branch in PROTECTED_BRANCHES:
        print(f"Direct pushes to the {current_branch} branch are not allowed.")
        print("Please use a pull request to merge changes.")
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
