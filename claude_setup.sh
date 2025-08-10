# Claude

# Download and install nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

# in lieu of restarting the shell
\. "$HOME/.nvm/nvm.sh"

# Download and install Node.js:
nvm install 22

# Verify the Node.js version:
node -v # Should print "v22.17.0".
nvm current # Should print "v22.17.0".

# Verify npm version:
npm -v # Should print "10.9.2".


npm install -g https://gaccode.com/claudecode/install --registry=https://registry.npmmirror.com

# Get npm global prefix and add to PATH
NPM_PREFIX=$(npm config get prefix)
export PATH="$NPM_PREFIX/bin:$PATH"

# Verify claude installation
which claude || echo "Claude not found in PATH, trying alternative locations..."

# export python path
export PYTHONPATH=/data3/Qwen2.5-VL-main

# Add to shell profile for permanent access
echo "Adding Claude to PATH permanently..."
if [[ -f "$HOME/.bashrc" ]]; then
    echo "export PATH=\"$NPM_PREFIX/bin:\$PATH\"" >> "$HOME/.bashrc"
    echo "Added to .bashrc"
fi