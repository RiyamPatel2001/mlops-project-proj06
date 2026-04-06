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
