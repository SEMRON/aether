"""
Prime Intellect API provider utilities for GPU pod provisioning.

This module provides functionality to interact with the Prime Intellect API
for provisioning, managing, and destroying GPU pods.
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


def _normalize_api_value(value: Any) -> Optional[str]:
    """
    Normalize API values that may be returned as lists.

    The Prime Intellect API sometimes returns string fields as single-element lists.
    This helper extracts the first element if it's a list, or returns the value as-is.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return value


def get_api_token() -> str:
    """Get the Prime Intellect API token from environment or prompt."""
    api_token = os.environ.get("PRIME_INTELLECT_API_TOKEN")
    if not api_token:
        # Fallback to the legacy env var name
        api_token = os.environ.get("api_token")
    if not api_token:
        print(
            "Please set PRIME_INTELLECT_API_TOKEN environment variable",
            file=sys.stderr,
        )
        raise RuntimeError("Missing PRIME_INTELLECT_API_TOKEN")
    return api_token


@dataclass
class SSHConnectionInfo:
    """Parsed SSH connection information."""

    user: str
    host: str
    port: int

    @classmethod
    def from_connection_string(
        cls, ssh_connection: str
    ) -> Optional["SSHConnectionInfo"]:
        """
        Parse SSH connection string like 'root@194.68.245.2 -p 22168'.

        Returns None if the string cannot be parsed.
        """
        if not ssh_connection:
            return None

        try:
            # Handle case where API returns a list instead of a string
            if isinstance(ssh_connection, list):
                if not ssh_connection:
                    return None
                ssh_connection = ssh_connection[0]

            parts = ssh_connection.split()
            user_host = parts[0]
            user, host = user_host.split("@")

            port = 22  # default
            if "-p" in parts:
                port_index = parts.index("-p") + 1
                if port_index < len(parts):
                    port = int(parts[port_index])

            return cls(user=user, host=host, port=port)
        except (ValueError, IndexError, AttributeError) as e:
            print(f"Error parsing SSH connection string: {e}", file=sys.stderr)
            return None


@dataclass
class PodStatus:
    """Status information for a Prime Intellect pod."""

    pod_id: str
    provider_type: str
    status: str
    ssh_connection: Optional[str]
    ip: Optional[str]
    installation_status: Optional[str]
    installation_failure: Optional[str]
    installation_progress: Optional[int]

    @property
    def ssh_info(self) -> Optional[SSHConnectionInfo]:
        """Parse and return SSH connection info."""
        if self.ssh_connection:
            return SSHConnectionInfo.from_connection_string(self.ssh_connection)
        return None

    @property
    def is_active(self) -> bool:
        return self.status == "ACTIVE"

    @property
    def is_provisioning(self) -> bool:
        return self.status == "PROVISIONING"


@dataclass
class PodInfo:
    """Information about an existing pod from the list API."""

    pod_id: str
    name: str
    provider_type: str
    gpu_name: str
    gpu_count: int
    price_hr: float
    status: str
    ssh_connection: Optional[str]
    ip: Optional[str]
    team_id: Optional[str]
    environment_type: Optional[str]
    created_at: Optional[str]
    raw_response: Dict[str, Any]

    @property
    def ssh_info(self) -> Optional[SSHConnectionInfo]:
        """Parse and return SSH connection info."""
        if self.ssh_connection:
            return SSHConnectionInfo.from_connection_string(self.ssh_connection)
        return None


@dataclass
class ProvisionedPod:
    """Response from provisioning a new pod."""

    pod_id: str
    name: str
    provider_type: str
    gpu_name: str
    gpu_count: int
    price_hr: float
    status: str
    ssh_connection: Optional[str]
    ip: Optional[str]
    cloud_id: str
    raw_response: Dict[str, Any]

    @property
    def ssh_info(self) -> Optional[SSHConnectionInfo]:
        """Parse and return SSH connection info."""
        if self.ssh_connection:
            return SSHConnectionInfo.from_connection_string(self.ssh_connection)
        return None


@dataclass
class AvailableInstance:
    """Information about an available GPU instance."""

    cloud_id: str
    gpu_type: str
    data_center: str
    country: str
    security: str
    socket: str
    provider: str
    images: List[str]
    on_demand_price: Optional[float]
    currency: str


class PrimeIntellectClient:
    """Client for interacting with the Prime Intellect API."""

    BASE_URL = "https://api.primeintellect.ai/api/v1"

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or get_api_token()
        self._last_response: Optional[Dict[str, Any]] = None
        self._last_status_code: int = 0

    def _get_headers(self, with_json: bool = False) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_token}"}
        if with_json:
            headers["Content-Type"] = "application/json"
        return headers

    def _print_curl_command(
        self,
        method: str,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print a curl command that can be used to replicate the request."""
        if json_data:
            json_str = json.dumps(json_data)
            # Escape single quotes in JSON for shell
            json_str_escaped = json_str.replace("'", "'\"'\"'")
            print(
                f"curl --request {method} --url '{url}' "
                f"--header \"Authorization: Bearer $PRIME_INTELLECT_API_TOKEN\" "
                f"--header 'Content-Type: application/json' "
                f"--data '{json_str_escaped}'",
                file=sys.stderr,
            )
        else:
            print(
                f"curl --request {method} --url '{url}' "
                f"--header \"Authorization: Bearer $PRIME_INTELLECT_API_TOKEN\"",
                file=sys.stderr,
            )

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make an API request."""
        url = f"{self.BASE_URL}{endpoint}"

        try:
            if json_data:
                response = requests.request(
                    method,
                    url,
                    headers=self._get_headers(with_json=True),
                    json=json_data,
                )
            else:
                response = requests.request(
                    method, url, headers=self._get_headers()
                )

            self._last_status_code = response.status_code

            try:
                self._last_response = response.json()
            except json.JSONDecodeError:
                self._last_response = {"raw": response.text}

            if not (200 <= response.status_code < 300):
                print(
                    "\n" + "=" * 60,
                    file=sys.stderr,
                )
                print(
                    f"API Request failed: {url}",
                    file=sys.stderr,
                )
                print(
                    f"HTTP Status: {response.status_code}",
                    file=sys.stderr,
                )
                print(
                    "\nResponse body:",
                    file=sys.stderr,
                )
                print(
                    json.dumps(self._last_response, indent=2),
                    file=sys.stderr,
                )
                print(
                    "\nTo replicate, run:",
                    file=sys.stderr,
                )
                self._print_curl_command(method, url, json_data)
                print(
                    "=" * 60 + "\n",
                    file=sys.stderr,
                )
                return None

            return self._last_response

        except requests.RequestException as e:
            print(
                "\n" + "=" * 60,
                file=sys.stderr,
            )
            print(f"Request failed: {url}", file=sys.stderr)
            print(f"Error: {e}", file=sys.stderr)
            print(
                "\nTo replicate, run:",
                file=sys.stderr,
            )
            self._print_curl_command(method, url, json_data)
            print(
                "=" * 60 + "\n",
                file=sys.stderr,
            )
            self._last_status_code = 0
            self._last_response = None
            return None

    def get_available_gpus(
        self,
        gpu_type: Optional[str] = None,
        regions: Optional[List[str]] = None,
        gpu_count: int = 1,
    ) -> List[AvailableInstance]:
        """
        Fetch available GPU instances.

        Args:
            gpu_type: Specific GPU type to filter by (e.g., 'A40_48GB')
            regions: List of regions to search
            gpu_count: Number of GPUs required

        Returns:
            List of available instances
        """
        params = [f"gpu_count={gpu_count}"]

        if gpu_type:
            params.append(f"gpu_type={gpu_type}")

        if regions:
            for region in regions:
                params.append(f"regions={region}")

        endpoint = f"/availability/gpus?{'&'.join(params)}"
        response = self._request("GET", endpoint)

        if not response:
            return []

        items = response.get("items", [])
        result = []

        for item in items:
            prices = item.get("prices", {})
            result.append(
                AvailableInstance(
                    cloud_id=item.get("cloudId", ""),
                    gpu_type=item.get("gpuType", ""),
                    data_center=item.get("dataCenter", ""),
                    country=item.get("country", ""),
                    security=item.get("security", ""),
                    socket=item.get("socket", ""),
                    provider=item.get("provider", ""),
                    images=item.get("images", []),
                    on_demand_price=prices.get("onDemand"),
                    currency=prices.get("currency", "USD"),
                )
            )

        return result

    def find_instance_by_cloud_id(
        self,
        cloud_id: str,
        gpu_count: int = 1,
        regions: Optional[List[str]] = None,
        os_prefix: str = "ubuntu_22",
    ) -> Tuple[Optional[AvailableInstance], Optional[str]]:
        """
        Find a specific instance by cloud ID.

        Args:
            cloud_id: The explicit cloud ID to find
            gpu_count: Number of GPUs required
            regions: List of regions to search
            os_prefix: Required OS image prefix

        Returns:
            Tuple of (instance, matching_image) or (None, None) if not found
        """
        if regions is None:
            regions = ["eu_west", "eu_north", "eu_east", "united_states"]

        # Fetch all available instances (no gpu_type filter)
        instances = self.get_available_gpus(
            gpu_type=None, regions=regions, gpu_count=gpu_count
        )

        for instance in instances:
            if instance.cloud_id == cloud_id:
                # Find matching image
                matching_image = None
                for image in instance.images:
                    if image.startswith(os_prefix):
                        matching_image = image
                        break

                # If no matching OS found, use first available image
                if not matching_image and instance.images:
                    matching_image = instance.images[0]
                    print(
                        f"Warning: No image matching '{os_prefix}' found for "
                        f"cloud_id {cloud_id}, using {matching_image}",
                        file=sys.stderr,
                    )

                if matching_image:
                    return instance, matching_image

        return None, None

    def find_matching_instances(
        self,
        gpu_type: str,
        gpu_count: int = 1,
        regions: Optional[List[str]] = None,
        os_prefix: str = "ubuntu_22",
        provider_blacklist: Optional[List[str]] = None,
    ) -> List[Tuple[AvailableInstance, str]]:
        """
        Find all available instances matching criteria, sorted by price.

        Args:
            gpu_type: GPU type to search for
            gpu_count: Number of GPUs required
            regions: List of regions to search
            os_prefix: Required OS image prefix
            provider_blacklist: List of providers to exclude

        Returns:
            List of (instance, matching_image) tuples, sorted by price ascending
        """
        if regions is None:
            regions = ["eu_west", "eu_north", "eu_east", "united_states"]

        if provider_blacklist is None:
            provider_blacklist = []

        instances = self.get_available_gpus(
            gpu_type=gpu_type, regions=regions, gpu_count=gpu_count
        )

        matching: List[Tuple[AvailableInstance, str, float]] = []

        blacklist_lower = [p.lower() for p in provider_blacklist]

        for instance in instances:
            # Skip blacklisted providers
            if instance.provider.lower() in blacklist_lower:
                continue

            # Find matching image
            matching_image = None
            for image in instance.images:
                if image.startswith(os_prefix):
                    matching_image = image
                    break

            if not matching_image:
                continue

            # Add to list with price for sorting
            price = instance.on_demand_price if instance.on_demand_price is not None else float("inf")
            matching.append((instance, matching_image, price))

        # Sort by price ascending
        matching.sort(key=lambda x: x[2])

        # Return without the price
        return [(inst, img) for inst, img, _ in matching]

    def find_best_instance(
        self,
        gpu_type: str,
        gpu_count: int = 1,
        regions: Optional[List[str]] = None,
        os_prefix: str = "ubuntu_22",
        provider_blacklist: Optional[List[str]] = None,
    ) -> Tuple[Optional[AvailableInstance], Optional[str]]:
        """
        Find the best (cheapest) available instance matching criteria.

        Args:
            gpu_type: GPU type to search for
            gpu_count: Number of GPUs required
            regions: List of regions to search
            os_prefix: Required OS image prefix
            provider_blacklist: List of providers to exclude

        Returns:
            Tuple of (best instance, matching image name) or (None, None)
        """
        matches = self.find_matching_instances(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            regions=regions,
            os_prefix=os_prefix,
            provider_blacklist=provider_blacklist,
        )

        if matches:
            return matches[0]
        return None, None

    def provision_pod(
        self,
        name: str,
        cloud_id: str,
        gpu_type: str,
        gpu_count: int,
        image: str,
        data_center_id: str,
        country: str,
        security: str,
        socket: str,
        provider: str,
        team_id: str,
        disk_size: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> Optional[ProvisionedPod]:
        """
        Provision a new GPU pod.

        Returns:
            ProvisionedPod on success, None on failure
        """
        return self._provision_pod_internal(
            name=name,
            cloud_id=cloud_id,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            image=image,
            data_center_id=data_center_id,
            country=country,
            security=security,
            socket=socket,
            provider=provider,
            team_id=team_id,
            disk_size=disk_size,
            env_vars=env_vars,
        )

    def _provision_pod_internal(
        self,
        name: str,
        cloud_id: str,
        gpu_type: str,
        gpu_count: int,
        image: str,
        data_center_id: str,
        country: str,
        security: str,
        socket: str,
        provider: str,
        team_id: str,
        disk_size: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> Optional[ProvisionedPod]:
        """
        Internal method to provision a pod (no retries).

        Returns:
            ProvisionedPod on success, None on failure
        """
        pod_config: Dict[str, Any] = {
            "name": name,
            "cloudId": cloud_id,
            "gpuType": gpu_type,
            "socket": socket,
            "gpuCount": gpu_count,
            "image": image,
            "dataCenterId": data_center_id,
            "country": country,
            "security": security,
        }

        if disk_size:
            pod_config["diskSize"] = disk_size

        if env_vars:
            pod_config["envVars"] = [
                {"key": k, "value": v} for k, v in env_vars.items()
            ]

        provision_data = {
            "pod": pod_config,
            "provider": {"type": provider},
            "team": {"teamId": team_id},
        }

        response = self._request("POST", "/pods/", provision_data)

        if not response:
            return None

        return ProvisionedPod(
            pod_id=response.get("id", ""),
            name=response.get("name", ""),
            provider_type=response.get("providerType", ""),
            gpu_name=response.get("gpuName", ""),
            gpu_count=response.get("gpuCount", 0),
            price_hr=response.get("priceHr", 0),
            status=response.get("status", ""),
            ssh_connection=response.get("sshConnection"),
            ip=response.get("ip"),
            cloud_id=cloud_id,
            raw_response=response,
        )

    def get_pod_statuses(
        self, pod_ids: List[str]
    ) -> Dict[str, Optional[PodStatus]]:
        """
        Get status for multiple pods.

        Args:
            pod_ids: List of pod IDs to query

        Returns:
            Dict mapping pod_id to PodStatus (or None if not found)
        """
        if not pod_ids:
            return {}

        params = "&".join([f"pod_ids={pid}" for pid in pod_ids])
        endpoint = f"/pods/status?{params}"

        response = self._request("GET", endpoint)

        if not response:
            return {pid: None for pid in pod_ids}

        data_items = response.get("data", [])

        result: Dict[str, Optional[PodStatus]] = {pid: None for pid in pod_ids}

        for item in data_items:
            pod_id = item.get("podId")
            if pod_id in result:
                result[pod_id] = PodStatus(
                    pod_id=pod_id,
                    provider_type=item.get("providerType", ""),
                    status=item.get("status", "UNKNOWN"),
                    ssh_connection=_normalize_api_value(item.get("sshConnection")),
                    ip=_normalize_api_value(item.get("ip")),
                    installation_status=item.get("installationStatus"),
                    installation_failure=item.get("installationFailure"),
                    installation_progress=item.get("installationProgress"),
                )

        return result

    def get_pod_status(self, pod_id: str) -> Optional[PodStatus]:
        """Get status for a single pod."""
        statuses = self.get_pod_statuses([pod_id])
        return statuses.get(pod_id)

    def list_all_pods(self, limit: int = 100) -> List[PodInfo]:
        """
        List all pods for the current user/team.

        Args:
            limit: Maximum number of pods to return

        Returns:
            List of PodInfo objects
        """
        endpoint = f"/pods/?limit={limit}"
        response = self._request("GET", endpoint)

        if not response:
            return []

        data_items = response.get("data", [])
        result = []

        for item in data_items:
            result.append(
                PodInfo(
                    pod_id=item.get("id", ""),
                    name=item.get("name", ""),
                    provider_type=item.get("providerType", ""),
                    gpu_name=item.get("gpuName", ""),
                    gpu_count=item.get("gpuCount", 0),
                    price_hr=item.get("priceHr", 0),
                    status=item.get("status", ""),
                    ssh_connection=_normalize_api_value(item.get("sshConnection")),
                    ip=_normalize_api_value(item.get("ip")),
                    team_id=item.get("teamId"),
                    environment_type=item.get("environmentType"),
                    created_at=item.get("createdAt"),
                    raw_response=item,
                )
            )

        return result

    def delete_pod(self, pod_id: str) -> bool:
        """
        Delete (terminate) a pod.

        Returns:
            True on success, False on failure
        """
        self._request("DELETE", f"/pods/{pod_id}")
        return self._last_status_code in (200, 204)

    def wait_for_pod_active(
        self,
        pod_id: str,
        poll_interval: int = 10,
        timeout: int = 300,
        verbose: bool = False,
    ) -> Optional[PodStatus]:
        """
        Wait for a pod to become active.

        Args:
            pod_id: Pod ID to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            verbose: Print progress updates

        Returns:
            PodStatus when active, None on timeout or error
        """
        import time

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                if verbose:
                    print(
                        f"Timeout waiting for pod {pod_id} to become active",
                        file=sys.stderr,
                    )
                return None

            status = self.get_pod_status(pod_id)

            if status is None:
                if verbose:
                    print(
                        f"Failed to get status for pod {pod_id}",
                        file=sys.stderr,
                    )
                return None

            if verbose:
                print(
                    f"Pod {pod_id}: status={status.status}, "
                    f"elapsed={int(elapsed)}s"
                )

            if status.is_active:
                return status

            if status.status in ("FAILED", "TERMINATED", "ERROR"):
                if verbose:
                    print(
                        f"Pod {pod_id} entered terminal state: {status.status}",
                        file=sys.stderr,
                    )
                return None

            time.sleep(poll_interval)

    def provision_pod_by_cloud_id(
        self,
        name: str,
        cloud_id: str,
        gpu_count: int,
        team_id: str,
        regions: Optional[List[str]] = None,
        os_prefix: str = "ubuntu_22",
        disk_size: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> Optional[ProvisionedPod]:
        """
        Provision a pod using an explicit cloud ID.

        Args:
            name: Name for the pod
            cloud_id: Explicit cloud ID to use
            gpu_count: Number of GPUs required
            team_id: Prime Intellect team ID
            regions: List of regions to search
            os_prefix: Required OS image prefix
            disk_size: Optional disk size in GB
            env_vars: Optional environment variables
            verbose: Print progress

        Returns:
            ProvisionedPod on success, None on failure
        """
        if verbose:
            print(f"Looking up cloud_id: {cloud_id}")

        instance, image = self.find_instance_by_cloud_id(
            cloud_id=cloud_id,
            gpu_count=gpu_count,
            regions=regions,
            os_prefix=os_prefix,
        )

        if not instance or not image:
            print(
                f"No instance found with cloud_id: {cloud_id}",
                file=sys.stderr,
            )
            return None

        if verbose:
            print(f"Found instance: {instance.cloud_id}")
            print(f"  GPU Type: {instance.gpu_type}")
            print(f"  Provider: {instance.provider}")
            print(f"  Region: {instance.country}")
            print(f"  Price: {instance.on_demand_price} {instance.currency}/hr")
            print(f"  Image: {image}")

        pod = self._provision_pod_internal(
            name=name,
            cloud_id=instance.cloud_id,
            gpu_type=instance.gpu_type,
            gpu_count=gpu_count,
            image=image,
            data_center_id=instance.data_center,
            country=instance.country,
            security=instance.security,
            socket=instance.socket,
            provider=instance.provider,
            team_id=team_id,
            disk_size=disk_size,
            env_vars=env_vars,
        )

        if pod and verbose:
            print(f"Successfully provisioned pod: {pod.pod_id}")

        return pod

    def provision_pod_with_fallback(
        self,
        name: str,
        gpu_type: str,
        gpu_count: int,
        team_id: str,
        regions: Optional[List[str]] = None,
        os_prefix: str = "ubuntu_22",
        provider_blacklist: Optional[List[str]] = None,
        disk_size: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        max_attempts: int = 5,
        verbose: bool = True,
        preferred_cloud_ids: Optional[List[str]] = None,
    ) -> Optional[ProvisionedPod]:
        """
        Provision a pod, trying multiple instances if provisioning fails.

        Args:
            name: Name for the pod
            gpu_type: GPU type to search for
            gpu_count: Number of GPUs required
            team_id: Prime Intellect team ID
            regions: List of regions to search
            os_prefix: Required OS image prefix
            provider_blacklist: List of providers to exclude
            disk_size: Optional disk size in GB
            env_vars: Optional environment variables
            max_attempts: Maximum number of instances to try
            verbose: Print progress
            preferred_cloud_ids: List of cloud IDs to try first (e.g., from sibling nodes in same array)

        Returns:
            ProvisionedPod on success, None if all attempts fail
        """
        matches = self.find_matching_instances(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            regions=regions,
            os_prefix=os_prefix,
            provider_blacklist=provider_blacklist,
        )

        if not matches:
            print(
                f"No available instances found for {gpu_type} "
                f"with OS prefix {os_prefix}",
                file=sys.stderr,
            )
            return None

        # Reorder matches to prioritize preferred cloud IDs
        if preferred_cloud_ids:
            preferred_set = set(preferred_cloud_ids)
            preferred_matches = []
            other_matches = []
            for match in matches:
                if match[0].cloud_id in preferred_set:
                    preferred_matches.append(match)
                else:
                    other_matches.append(match)
            matches = preferred_matches + other_matches
            if verbose and preferred_matches:
                print(f"Prioritizing {len(preferred_matches)} instances matching sibling cloud IDs")

        if verbose:
            print(f"Found {len(matches)} matching instances, will try up to {max_attempts}")

        attempts = min(max_attempts, len(matches))

        for i, (instance, image) in enumerate(matches[:attempts]):
            if verbose:
                print(
                    f"\nAttempt {i + 1}/{attempts}: {instance.cloud_id} "
                    f"({instance.provider}, {instance.country}, "
                    f"{instance.on_demand_price} {instance.currency}/hr)"
                )

            pod = self._provision_pod_internal(
                name=name,
                cloud_id=instance.cloud_id,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                image=image,
                data_center_id=instance.data_center,
                country=instance.country,
                security=instance.security,
                socket=instance.socket,
                provider=instance.provider,
                team_id=team_id,
                disk_size=disk_size,
                env_vars=env_vars,
            )

            if pod:
                if verbose:
                    print(f"Successfully provisioned pod: {pod.pod_id}")
                return pod

            if verbose:
                print("Provisioning failed, trying next instance...")

        print(
            f"All {attempts} provisioning attempts failed",
            file=sys.stderr,
        )
        return None


def provision_pod_simple(
    name: str,
    gpu_type: str,
    team_id: str,
    gpu_count: int = 1,
    regions: Optional[List[str]] = None,
    os_prefix: str = "ubuntu_22",
    provider_blacklist: Optional[List[str]] = None,
    wait_for_active: bool = True,
    verbose: bool = False,
) -> Optional[Tuple[ProvisionedPod, Optional[PodStatus]]]:
    """
    High-level function to provision a pod with sensible defaults.

    Args:
        name: Name for the pod
        gpu_type: GPU type (e.g., 'A40_48GB')
        team_id: Prime Intellect team ID
        gpu_count: Number of GPUs
        regions: Regions to search (default: EU and US)
        os_prefix: OS image prefix (default: ubuntu_22)
        provider_blacklist: Providers to exclude (default: ['runpod'])
        wait_for_active: Wait for pod to become active before returning
        verbose: Print progress

    Returns:
        Tuple of (ProvisionedPod, PodStatus) on success, None on failure
    """
    if provider_blacklist is None:
        provider_blacklist = ["runpod"]

    client = PrimeIntellectClient()

    if verbose:
        print(f"Searching for {gpu_count}x {gpu_type} instances...")

    instance, image = client.find_best_instance(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        regions=regions,
        os_prefix=os_prefix,
        provider_blacklist=provider_blacklist,
    )

    if not instance or not image:
        print(
            f"No available instance found for {gpu_type} "
            f"with OS prefix {os_prefix}",
            file=sys.stderr,
        )
        return None

    if verbose:
        print(f"Found instance: {instance.cloud_id}")
        print(f"  Provider: {instance.provider}")
        print(f"  Region: {instance.country}")
        print(f"  Price: {instance.on_demand_price} {instance.currency}/hr")
        print(f"  Image: {image}")

    pod = client.provision_pod(
        name=name,
        cloud_id=instance.cloud_id,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        image=image,
        data_center_id=instance.data_center,
        country=instance.country,
        security=instance.security,
        socket=instance.socket,
        provider=instance.provider,
        team_id=team_id,
    )

    if not pod:
        print("Failed to provision pod", file=sys.stderr)
        return None

    if verbose:
        print(f"Pod provisioned: {pod.pod_id}")
        print(f"  Status: {pod.status}")

    final_status: Optional[PodStatus] = None

    if wait_for_active:
        if verbose:
            print("Waiting for pod to become active...")

        final_status = client.wait_for_pod_active(
            pod.pod_id, verbose=verbose
        )

        if final_status and verbose:
            print("Pod is active!")
            if final_status.ssh_info:
                info = final_status.ssh_info
                print(f"  SSH: ssh -p {info.port} {info.user}@{info.host}")

    return pod, final_status
