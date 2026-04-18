output "floating_ip" {
  description = "Public IP of the VM"
  value       = openstack_networking_floatingip_v2.fip.address
}

output "ssh_command" {
  value = "ssh cc@${openstack_networking_floatingip_v2.fip.address}"
}

output "fastapi_docs_url" {
  value = "http://${openstack_networking_floatingip_v2.fip.address}:8000/docs"
}

output "actual_budget_url" {
  value = "http://${openstack_networking_floatingip_v2.fip.address}:5006"
}
