{
  description = "Python development environment with uv venv creation and automatic activation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312; # Define the python interpreter
        pythonPackages = python.pkgs;


        torchtext = pythonPackages.buildPythonPackage rec {
          pname = "torchtext";
          version = "0.18.0";

          src = pkgs.fetchFromGitHub {
            owner = "pytorch";
            repo = "text";
            rev = "v${version}";
            fetchSubmodules = true;
            deepClone = true;
            sha256 = "sha256-ok91rw/76ivtTTd3DkdDG7N2aZE5WqPuZE4YbbQ0pYU=";
          };

          dontUseCmakeConfigure = true;

          doCheck = false;

          propagatedBuildInputs = [
            pythonPackages.torch
            pythonPackages.numpy
            pythonPackages.requests
            pythonPackages.tqdm
          ];

          nativeBuildInputs = [
            pkgs.cmake
            pkgs.ninja
            pkgs.which
            pkgs.stdenv.cc
            pkgs.git
          ];

          buildInputs = [
            pythonPackages.pybind11
          ];

          checkInputs = [
            pythonPackages.pytestCheckHook
          ];

          pythonImportsCheck = [ "torchtext" ];

           postPatch = ''
            echo "Patching setup.py to prevent internal submodule initialization..."
            sed -i '/^[[:space:]]*_init_submodule()/s/^/# /' setup.py
            echo "Patching setup.py to set ROOT_DIR correctly using NIX_PROJECT_ROOT_DIR..."
            substituteInPlace setup.py \
              --replace "ROOT_DIR = Path(__file__).parent.resolve()" \
                        "import os; ROOT_DIR = Path(os.environ.get('NIX_PROJECT_ROOT_DIR', Path(__file__).parent.resolve()))"
            echo "Finished patching setup.py."
          '';


          preBuild = ''
  EXPECTED_PROJECT_ROOT="$NIX_BUILD_TOP/source" # Assuming "source" is the dir after unpack
  
  echo "Current directory before potential cd in preBuild: $(pwd)"
  if [ "$(pwd)" != "$EXPECTED_PROJECT_ROOT" ]; then
    echo "WARNING: Current working directory is $(pwd), but expected $EXPECTED_PROJECT_ROOT."
    echo "Attempting to change directory to $EXPECTED_PROJECT_ROOT."
    cd "$EXPECTED_PROJECT_ROOT"
    if [ "$(pwd)" != "$EXPECTED_PROJECT_ROOT" ]; then
      echo "FATAL ERROR: Failed to change CWD to $EXPECTED_PROJECT_ROOT. Current CWD is $(pwd)."
      exit 1
    fi
    echo "Successfully changed CWD to $(pwd)."
  else
    echo "Current working directory is already $EXPECTED_PROJECT_ROOT."
  fi
  
  export NIX_PROJECT_ROOT_DIR=$(pwd) # This will be used by the patched setup.py
  echo "NIX_PROJECT_ROOT_DIR successfully set to: $NIX_PROJECT_ROOT_DIR"
'';

        };

        lib-path = with pkgs; lib.makeLibraryPath [
          libffi
          openssl
          stdenv.cc.cc
        ];

        flakeContentFile = pkgs.writeTextFile {
          name = "flake-template";
          text = builtins.readFile ./flake.nix;
        };

        installScript = pkgs.writeScriptBin "install-template" ''
          #!${pkgs.bash}/bin/bash
          set -e # Exit on error

          if [ -f flake.nix ]; then
            echo "flake.nix already exists in current directory. Aborting."
            exit 1
          fi

          echo "Installing Python flake template..."
          # Copy flake content from the template file to make it writable
          cp ${flakeContentFile} ./flake.nix
          chmod +w ./flake.nix

          # Create .gitignore if it doesn't exist
          if [ ! -f .gitignore ]; then
            echo "Creating .gitignore..."
            cat > .gitignore << 'EOF'
.venv/
.idea/
__pycache__/
env/
result
**/*.pyc

# DL related
data/
wandb/
outputs/

EOF
          fi

          echo "Template installed successfully!"
          echo "You can now use 'nix develop' in this directory."
        '';

        venvDir = ".venv";

      in
        {
        devShells.default = pkgs.mkShell {
          venvDir = venvDir; # directory to be activated automatically
          packages = with pkgs; [
            python 

            pythonPackages.matplotlib
            pythonPackages.numpy

            torchtext
            pythonPackages.torch
            pythonPackages.torchvision
            pythonPackages.datasets
            pythonPackages.transformers
            pythonPackages.tokenizers

            pythonPackages.nltk
            pythonPackages.ipykernel

            # Tools
            pythonPackages.venvShellHook 

            git
            stdenv.cc.cc.lib
            rsync
            bashInteractive 
            installScript 
            coreutils 
            cmake
          ];

          buildInputs = with pkgs; [
            cmake
            openssl
            zlib
          ];

          # This hook runs BEFORE the interactive shell starts.
          # Its job now is to PREPARE the environment.
          shellHook = ''
            set -e 

            # Set SOURCE_DATE_EPOCH 
            SOURCE_DATE_EPOCH=$(date +%s)

            # Ensure system libraries are findable
            export LD_LIBRARY_PATH="${lib-path}:$LD_LIBRARY_PATH"

            # Use the variable defined in 'let' block
            VENV_DIR="${venvDir}" 

            # Check if the venv directory needs to be created
            if [ ! -d "$VENV_DIR" ]; then
            echo "Creating Python virtual environment using uv in $VENV_DIR..."
            # Create the venv using uv, pointing to the Nix-provided Python
            ${pkgs.uv}/bin/uv venv -p ${python}/bin/python3.12 "$VENV_DIR" 
            echo "Virtual environment created."
            else
            echo "Using existing virtual environment in $VENV_DIR"
            fi

            set +e 

            if [ -f requirements.txt ]; then
            echo "Attempting to sync environment with requirements.txt using uv..."
            # Temporarily activate for the install command within this hook
            # Ensure the activate script exists before sourcing
            if [ -f "$VENV_DIR/bin/activate" ]; then
            source "$VENV_DIR/bin/activate"
            else
            echo "WARNING: Activation script $VENV_DIR/bin/activate not found. Skipping package sync."
            SKIP_SYNC=1 
            fi

            # Only run sync if activation succeeded
            if [ -z "$SKIP_SYNC" ]; then
            ${pkgs.uv}/bin/uv pip sync requirements.txt
            SYNC_EXIT_CODE=$? # Capture exit code

            # Deactivate the temporary activation within the hook
            # Check if deactivate function exists before calling (robustness)
            if command -v deactivate > /dev/null; then
            deactivate
            fi

            if [ $SYNC_EXIT_CODE -ne 0 ]; then
            echo ""
            echo "WARNING: 'uv pip sync requirements.txt' failed (exit code $SYNC_EXIT_CODE)."
            echo "Your Nix shell is ready, but Python packages may be missing or incorrect."
            echo "Check error messages above, requirements.txt, and network connection."
            echo "You might need to run 'uv pip sync requirements.txt' manually."
            echo ""
            else
            echo "Environment sync with requirements.txt successful."
            fi
            fi # End check for SKIP_SYNC
            else
            echo "No requirements.txt found. Skipping package installation/sync."
            fi

            if [ ! -f flake.nix ]; then
            echo "---------------------------------------------------------------------"
            echo "This is a temporary shell. To make it persistent for this project,"
            echo "run 'install-template' to copy the flake.nix file here."
            echo "---------------------------------------------------------------------"
            fi

            echo "Nix shell environment configured. venvShellHook will now activate $VENV_DIR."
            '';

          postShellHook = ''
            VENV_DIR="${venvDir}" 

            # Ensure the target directory exists before creating symlinks
            mkdir -p ./$VENV_DIR/lib/python3.12/site-packages/

            # Symlink Nix store site-packages into the venv's site-packages
            ln -sfn ${python.sitePackages}/* ./$VENV_DIR/lib/python3.12/site-packages/
            '';
        };

        packages.default = installScript;
      }
    );
}
