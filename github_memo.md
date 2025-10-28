git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "complete environment"
git push origin main