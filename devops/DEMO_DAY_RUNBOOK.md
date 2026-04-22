# Demo Day Runbook — proj06

## Step 1: Create Chameleon reservation
- Go to chi.tacc.chameleoncloud.org → Reservations → Leases
- Create lease for 5 KVM nodes (m1.xlarge or equivalent)
- Note the reservation UUID

## Step 2: Provision infrastructure
```bash
cd devops/iac/tf
# Create terraform.tfvars:
# suffix     = "proj06"
# reservation = "<YOUR-RESERVATION-UUID>"
# key        = "id_rsa_chameleon"

terraform init
terraform apply
# Note the floating IP output
```

## Step 3: Update Ansible inventory
Edit devops/iac/ansible/k8s/inventory/mycluster/hosts.yaml
Replace FLOATING_IP_HERE with the actual floating IP from terraform output

## Step 4: Provision k3s cluster
```bash
cd devops/iac/ansible
# Run k3s install playbook (kubespray or k3s-ansible)
ansible-playbook -i k8s/inventory/mycluster/hosts.yaml k8s/cluster.yml \
  --private-key ~/.ssh/id_rsa_chameleon
```

## Step 5: Install ArgoCD
```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@<FLOATING-IP>
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

## Step 6: Apply ArgoCD apps (deploys everything from git)
```bash
kubectl apply -f devops/k8s/argocd/app-platform.yaml
kubectl apply -f devops/k8s/argocd/app-serving.yaml
# ArgoCD will deploy all manifests automatically
```

## Step 7: Write layer1_registry.json to serving PVC
Models are already in MinIO (persistent). Just write the registry:
```bash
kubectl run registry-init --image=busybox --restart=Never \
  --overrides='{"spec":{"nodeSelector":{"kubernetes.io/hostname":"proj06-node3"},"volumes":[{"name":"m","persistentVolumeClaim":{"claimName":"serving-models-pvc"}}],"containers":[{"name":"b","image":"busybox","command":["sh","-c","sleep 3600"],"volumeMounts":[{"name":"m","mountPath":"/mnt"}]}]}}' -n mlops

# Wait for pod, then copy registry file
kubectl cp devops/layer1_registry.json mlops/registry-init:/mnt/layer1_registry.json
kubectl delete pod registry-init -n mlops
```

## Step 8: Verify everything is up
```bash
kubectl get pods -n mlops
curl http://<FLOATING-IP>:30508/health | python3 -m json.tool
```

## Key credentials
- Grafana: admin / mlops1234  → http://<FLOATING-IP>:30030
- MLflow:  no auth           → http://<FLOATING-IP>:30500
- MinIO:   minioadmin / minioadmin → http://<FLOATING-IP>:30901
- ArgoCD:  admin / (kubectl get secret argocd-initial-admin-secret -n argocd -o jsonpath="{.data.password}" | base64 -d)
- ActualBudget: admin@admin.com / mlops1234 → http://<FLOATING-IP>:30506

## MLflow run IDs (models in MinIO — these survive cluster teardown)
- good  (minilm):       b8f1ad8433b7492e82726429df5b66a0
- fast  (fasttext):     5af29fbaa4b04abc9d21b18a19ccc736
- cheap (tfidf_logreg): bd19d31c1aa94500a0fa3a4f2cee4c94
