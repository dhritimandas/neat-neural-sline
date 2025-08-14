#!/bin/bash

echo "========================================"
echo "GitHub Repository Setup Instructions"
echo "========================================"
echo ""
echo "Step 1: Create a new repository on GitHub"
echo "----------------------------------------"
echo "1. Go to: https://github.com/new"
echo "2. Repository name: neat-neural-slime"
echo "3. Description: 'NEAT AI training system for SlimeVolley - Expert-level gameplay through staged learning'"
echo "4. Set to Public or Private as desired"
echo "5. DON'T initialize with README (we already have one)"
echo "6. Click 'Create repository'"
echo ""
echo "Step 2: Copy your repository URL"
echo "----------------------------------------"
echo "It will look like:"
echo "  HTTPS: https://github.com/YOUR_USERNAME/neat-neural-slime.git"
echo "  SSH:   git@github.com:YOUR_USERNAME/neat-neural-slime.git"
echo ""
read -p "Enter your GitHub repository URL: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "No URL provided. Exiting."
    exit 1
fi

echo ""
echo "Step 3: Adding remote and pushing"
echo "----------------------------------------"

# Add remote
git remote add origin "$REPO_URL"

# Verify remote was added
echo "Remote added:"
git remote -v

echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "========================================"
echo "✅ Repository setup complete!"
echo "========================================"
echo ""
echo "Your repository is now available at:"
echo "$REPO_URL"
echo ""
echo "Next steps:"
echo "1. Add a LICENSE file if desired"
echo "2. Set up GitHub Actions for automated testing"
echo "3. Add badges to README for build status"
echo "4. Enable GitHub Pages for documentation"
echo ""
echo "To clone on another machine:"
echo "git clone $REPO_URL"