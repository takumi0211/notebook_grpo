git fetch --all --prune
git reset --hard origin/main
git clean -fd

git add -A
git commit -m "upload first"
git push origin main