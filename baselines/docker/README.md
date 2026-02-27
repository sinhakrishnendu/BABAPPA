# Baseline Container Images

Pinned baseline images used by BABAPPA adapters:

- codeml (PAML): `quay.io/biocontainers/paml:4.10.7--hdfd78af_0`
- HyPhy: `stevenweaver/hyphy:2.5.63`

Optional local rebuild tags:

```bash
docker build -f baselines/docker/Dockerfile.codeml -t babappa/codeml:4.10.7 baselines/docker
docker build -f baselines/docker/Dockerfile.hyphy -t babappa/hyphy:2.5.63 baselines/docker
```

Force adapter backend:

```bash
export BABAPPA_CODEML_BACKEND=docker
export BABAPPA_HYPHY_BACKEND=docker
```
