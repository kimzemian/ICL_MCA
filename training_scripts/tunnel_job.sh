#!/bin/bash


#SBATCH --job-name="tunnel1"
#SBATCH --time=7-00:00
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH -n 8
#SBATCH --mem=100gb
#SBATCH --partition=default_partition
#SBATCH --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03


set -e

# add any required module loads here, e.g. a specific Python


CLI_PATH="${HOME}/vscode_cli"


# Install the VS Code CLI command if it doesn't exist
if [[ ! -e ${CLI_PATH}/code ]]; then
        echo "Downloading and installing the VS Code CLI command"
        mkdir -p "${HOME}/vscode_cli"
        pushd "${HOME}/vscode_cli"
        # Process from: https://code.visualstudio.com/docs/remote/tunnels#_using-the-code_cli
        curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
        # unpack the code binary file
        tar -xf vscode_cli.tar.gz
        # clean-up
        rm vscode_cli.tar.gz
        popd
fi

find "${HOME}/.vscode-server" -name "*.lock" -delete 2>/dev/null || true
find "${HOME}/.vscode/cli" -name "*.lock" -delete 2>/dev/null || true
${CLI_PATH}/code tunnel --accept-server-license-terms