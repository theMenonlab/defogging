# Local path audit

The staging pass removed known private workstation and CHPC path defaults from public-facing documentation and active helper scripts.

Before upload, run:

```bash
rg -n "/home/|/mnt/|/media/|/scratch|@|u[0-9]{7}" .
```

Expected remaining matches should be reviewed case by case. Result CSVs should use dataset-relative paths where possible.
