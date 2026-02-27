Using singularity

1. Pull an docker image
```
singularity pull docker://luminoctum/ubuntu22.04-cuda12.9-py3.10-canoe:2026-02-23
```

2. Check cache
```
singularity cache list
```

3. Clean cache
```
singularity cache clean
```

4. Set cache directory
```
export SINGULARITY_CACHEDIR=/path/to/cache
```

5. Run a singularity shell
```
singularity shell --nv ubuntu22.04-cuda12.9-py3.10-canoe_2026-02-23.sif
```
