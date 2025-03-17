# Local Development Cluster

## Prerequisites

* k3d
* kubectl
* docker
* helm
* tilt

### Steps
1. Create a local registry for pushing images directly from your machine

```sh
k3d registry create --port localhost:5111
```
2. Create the cluster and bind the registry

```sh
k3d cluster create grazer \
  --registry-use k3d-registry:5111 \
  --api-port 6550 \
  --servers 1 \
  --port 443:443@loadbalancer \
  --port 80:80@loadbalancer \
  --k3s-arg "--disable=traefik@server:0" \
  --wait \
  --verbose
```

  --k3s-arg "--kubelet-arg=eviction-hard=imagefs.available<1%,nodefs.available<1%" \
  --k3s-arg "--kubelet-arg=eviction-minimum-reclaim=imagefs.available=1%,nodefs.available=1%" \