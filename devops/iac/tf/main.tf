# Security groups for proj06 K8s cluster
resource "openstack_networking_secgroup_v2" "k8s_api" {
  name        = "allow-6443-${var.suffix}"
  description = "K8s API server"
}

resource "openstack_networking_secgroup_rule_v2" "k8s_api_rule" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 6443
  port_range_max    = 6443
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.k8s_api.id
}

resource "openstack_networking_secgroup_v2" "mlflow" {
  name        = "allow-30500-${var.suffix}"
  description = "MLflow NodePort"
}

resource "openstack_networking_secgroup_rule_v2" "mlflow_rule" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30500
  port_range_max    = 30500
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.mlflow.id
}

resource "openstack_networking_secgroup_v2" "actualbudget" {
  name        = "allow-30506-${var.suffix}"
  description = "ActualBudget NodePort"
}

resource "openstack_networking_secgroup_rule_v2" "actualbudget_rule" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30506
  port_range_max    = 30506
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.actualbudget.id
}

resource "openstack_networking_secgroup_v2" "serving" {
  name        = "allow-30508-${var.suffix}"
  description = "Transaction Classifier NodePort"
}

resource "openstack_networking_secgroup_rule_v2" "serving_rule" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30508
  port_range_max    = 30508
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.serving.id
}

resource "openstack_networking_secgroup_v2" "minio_console" {
  name        = "allow-30901-${var.suffix}"
  description = "MinIO Console NodePort"
}

resource "openstack_networking_secgroup_rule_v2" "minio_console_rule" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30901
  port_range_max    = 30901
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.minio_console.id
}

resource "openstack_networking_secgroup_v2" "minio_api" {
  name        = "allow-30900-${var.suffix}"
  description = "MinIO API NodePort"
}

resource "openstack_networking_secgroup_rule_v2" "minio_api_rule" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30900
  port_range_max    = 30900
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.minio_api.id
}

# Private network (no port security — all inter-node traffic allowed)
resource "openstack_networking_network_v2" "private_net" {
  name                  = "private-net-mlops-${var.suffix}"
  port_security_enabled = false
}

resource "openstack_networking_subnet_v2" "private_subnet" {
  name       = "private-subnet-mlops-${var.suffix}"
  network_id = openstack_networking_network_v2.private_net.id
  cidr       = "192.168.1.0/24"
  no_gateway = true
}

resource "openstack_networking_port_v2" "private_net_ports" {
  for_each              = var.nodes
  name                  = "port-${each.key}-mlops-${var.suffix}"
  network_id            = openstack_networking_network_v2.private_net.id
  port_security_enabled = false

  fixed_ip {
    subnet_id  = openstack_networking_subnet_v2.private_subnet.id
    ip_address = each.value
  }
}

resource "openstack_networking_port_v2" "sharednet1_ports" {
  for_each   = var.nodes
  name       = "sharednet1-${each.key}-mlops-${var.suffix}"
  network_id = data.openstack_networking_network_v2.sharednet1.id
  security_group_ids = [
    data.openstack_networking_secgroup_v2.allow_ssh.id,
    openstack_networking_secgroup_v2.k8s_api.id,
    openstack_networking_secgroup_v2.mlflow.id,
    openstack_networking_secgroup_v2.actualbudget.id,
    openstack_networking_secgroup_v2.serving.id,
    openstack_networking_secgroup_v2.minio_console.id,
    openstack_networking_secgroup_v2.minio_api.id,
    openstack_networking_secgroup_v2.grafana.id,
    openstack_networking_secgroup_v2.prometheus.id,
    openstack_networking_secgroup_v2.argocd.id,
    openstack_networking_secgroup_v2.adminer.id,
  ]
}

resource "openstack_compute_instance_v2" "nodes" {
  for_each = var.nodes

  name       = "${each.key}-mlops-${var.suffix}"
  image_name = "CC-Ubuntu24.04"
  flavor_id  = var.reservation
  key_pair   = var.key

  network {
    port = openstack_networking_port_v2.sharednet1_ports[each.key].id
  }

  network {
    port = openstack_networking_port_v2.private_net_ports[each.key].id
  }

  user_data = <<-EOF
    #! /bin/bash
    sudo echo "127.0.1.1 ${each.key}-mlops-${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}

resource "openstack_networking_floatingip_v2" "floating_ip" {
  pool        = "public"
  description = "MLOps proj06 floating IP"
  port_id     = openstack_networking_port_v2.sharednet1_ports["node1"].id
}

# Extra block volume for Docker image storage on node1
# Prevents disk pressure from large training images filling the boot disk
resource "openstack_blockstorage_volume_v3" "docker_storage" {
  name = "proj06-docker-storage"
  size = 50
}

resource "openstack_compute_volume_attach_v2" "docker_storage_attach" {
  instance_id = openstack_compute_instance_v2.nodes["node1"].id
  volume_id   = openstack_blockstorage_volume_v3.docker_storage.id
}

resource "openstack_networking_secgroup_v2" "grafana" {
  name        = "allow-30030-${var.suffix}"
  description = "Grafana NodePort"
}

resource "openstack_networking_secgroup_rule_v2" "grafana_rule" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30030
  port_range_max    = 30030
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.grafana.id
}

resource "openstack_networking_secgroup_v2" "prometheus" {
  name        = "allow-30090-${var.suffix}"
  description = "Prometheus NodePort"
}

resource "openstack_networking_secgroup_rule_v2" "prometheus_rule" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30090
  port_range_max    = 30090
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.prometheus.id
}

resource "openstack_networking_secgroup_v2" "adminer" {
  name        = "allow-30081-${var.suffix}"
  description = "Adminer NodePort"
}

resource "openstack_networking_secgroup_rule_v2" "adminer_rule" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30081
  port_range_max    = 30081
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.adminer.id
}

resource "openstack_networking_secgroup_v2" "argocd" {
  name        = "allow-30808-30809-${var.suffix}"
  description = "ArgoCD NodePorts"
}

resource "openstack_networking_secgroup_rule_v2" "argocd_http_rule" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 30808
  port_range_max    = 30809
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.argocd.id
}
