# GitHub Upload Notes

Use Git from the command line for this repository. The browser uploader can reject folders with more than 100 files even when the folder is small.

## Commit The Results And Paper Folders

```bash
cd /home/al/Documents/fog_imager/defogging_github_upload_20260615
git status --short
git add results paper
git commit -m "Add computational defogging result tables and paper files"
git push origin main
```

## Commit Documentation And Inference Updates

```bash
cd /home/al/Documents/fog_imager/defogging_github_upload_20260615
git status --short
git add README.md code data docs models requirements.txt
git commit -m "Improve computational defogging reproducibility docs"
git push origin main
```

If Git identity is missing, set it once:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```
