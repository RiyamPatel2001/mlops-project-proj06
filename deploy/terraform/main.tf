# ── Security Group ────────────────────────────────────────────────────────────

resource "openstack_networking_secgroup_v2" "serving_sg" {
  name        = "${var.instance_name}-sg"
  description = "Allow SSH, FastAPI (8000), Adminer (8080), Actual Budget (5006)"
}

resource "openstack_networking_secgroup_rule_v2" "ssh" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.serving_sg.id
}

resource "openstack_networking_secgroup_rule_v2" "fastapi" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8000
  port_range_max    = 8000
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.serving_sg.id
}

resource "openstack_networking_secgroup_rule_v2" "actual_budget" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 5006
  port_range_max    = 5006
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.serving_sg.id
}

resource "openstack_networking_secgroup_rule_v2" "adminer" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8080
  port_range_max    = 8080
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.serving_sg.id
}

# ── Floating IP ──────────────────────────────────────────────────────────────

resource "openstack_networking_floatingip_v2" "fip" {
  pool = "public"
}

# ── VM Instance ──────────────────────────────────────────────────────────────

resource "openstack_compute_instance_v2" "serving" {
  name        = var.instance_name
  image_name  = var.image_name
  flavor_name = "reservation:${var.reservation_id}"
  key_pair    = var.key_pair_name

  network {
    name = var.network_name
  }

  security_groups = [openstack_networking_secgroup_v2.serving_sg.name]

  user_data = templatefile("${path.module}/cloud-init.yaml", {
    postgres_dsn                = var.postgres_dsn
    mlflow_tracking_uri         = var.mlflow_tracking_uri
    mlflow_model_uri            = var.mlflow_model_uri
    embedding_service_url       = var.embedding_service_url
    layer1_model_kind           = var.layer1_model_kind
    layer1_mlflow_run_id        = var.layer1_mlflow_run_id
    layer1_mlflow_artifact_path = var.layer1_mlflow_artifact_path
    repo_url                    = var.repo_url
    repo_ref                    = var.repo_ref
    actual_build_node_memory_mb = var.actual_build_node_memory_mb
    docker_platform             = var.docker_platform
  })
}

# ── Attach floating IP ───────────────────────────────────────────────────────

resource "openstack_compute_floatingip_associate_v2" "fip" {
  floating_ip = openstack_networking_floatingip_v2.fip.address
  instance_id = openstack_compute_instance_v2.serving.id
}
