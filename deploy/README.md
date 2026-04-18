# Chameleon Cloud Deployment

This deployment layout is aimed at an Intel-based Chameleon KVM VM such as `m1.large` (4 vCPU, 8 GB RAM).

## Recommended path

Build on the VM itself from the checked-out source tree. That avoids the Apple Silicon vs Intel mismatch entirely.

```bash
git clone https://github.com/jayrajpamnani/MLOps-Serving.git
cd MLOps-Serving/deploy
cp .env.example .env
nano .env
./setup.sh
```

The setup script will:

- install Docker if needed
- create a 4 GB swapfile if the VM has no swap
- build `linux/amd64` images
- reuse `actual-custom:latest` automatically if you already loaded a prebuilt image

## If you want to build on your Mac and copy the image

If your laptop is Apple Silicon, build an AMD64 image explicitly:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f actual/sync-server.Dockerfile \
  -t actual-custom:latest \
  --load \
  actual

docker save actual-custom:latest | gzip > actual-custom-amd64.tar.gz
scp actual-custom-amd64.tar.gz cc@<FLOATING_IP>:~/
ssh cc@<FLOATING_IP> 'gunzip -c ~/actual-custom-amd64.tar.gz | sudo docker load'
ssh cc@<FLOATING_IP> 'cd ~/MLOps-Serving/deploy && ./setup.sh'
```

Once the AMD64-tagged image is loaded on the VM, `./setup.sh` will skip rebuilding the custom Actual image and only build the serving image.

## Terraform notes

For Chameleon KVM reservations, the instance should be launched with the reserved flavor name `reservation:<reservation_id>`, not by passing the reservation UUID as `flavor_id`.

The Terraform in `deploy/terraform/` has been adjusted for that and now:

- launches the VM with `flavor_name = "reservation:<reservation_id>"`
- clones the repo on first boot
- writes `deploy/.env`
- runs `deploy/setup.sh` from cloud-init

## Runtime endpoints

After deployment:

- FastAPI docs: `http://<FLOATING_IP>:8000/docs`
- Actual Budget: `http://<FLOATING_IP>:5006`

## Using cluster-internal PostgreSQL from a separate lease

If your serving VM is outside the shared Kubernetes lease, do not expose
PostgreSQL publicly. Instead, tunnel it through the cluster-access VM:

1. On the serving VM, run an SSH session to the cluster VM over the private
   `10.56.0.0/22` network.
2. In that SSH session, run `kubectl -n mlops port-forward` on the cluster VM.
3. Point `POSTGRES_DSN` at `host.docker.internal:<local_port>` so the container
   reaches the host-side tunnel instead of trying `localhost` inside the
   container.

This keeps PostgreSQL bound to loopback on both machines and avoids changing
cluster services or publishing the database on the public internet.

## Browser DB viewer

Adminer can run on the serving VM at:

```text
http://129.114.25.161:8080
```

Adminer login values for this setup:

- System: `PostgreSQL`
- Server: `host.docker.internal:15432`
- Username: `mlops_user`
- Password: `mlops_pass`
- Database: `mlops`
