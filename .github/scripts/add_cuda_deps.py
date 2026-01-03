#!/usr/bin/env python3
# Sigh1 pyproject.toml does not provide nice support for dynamic dependencies;
# But we want cuda-12.8 built packages to require 12.8, and 13.0 to require 13.0.
# Seems reasonable, but apparently too much of a niche case for PyPy ?!
# So the solution is to edit the pyproject.toml as part of the build script in the
# workflow :(
import subprocess
import sys
import tomlkit

def add_dependencies(deps_str, cuda_tag=None):
    # Parse dependencies from input
    deps = [dep.strip() for dep in deps_str.strip().split('\n') if dep.strip()]

    with open('pyproject.toml', 'r') as f:
        data = tomlkit.load(f)

    # Add CUDA dependencies
    for dep in deps:
        data['project']['dependencies'].append(dep)

    # Add CUDA tag to version if provided
    if cuda_tag:
        # Version may be dynamic (computed by setuptools-scm), so we need to get it
        if 'version' in data['project']:
            current_version = data['project']['version']
        else:
            # First check if we're exactly on a tag
            result = subprocess.run(
                ['git', 'describe', '--tags', '--exact-match', '--match', 'v*'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                # We're on an exact tag - use it directly
                current_version = result.stdout.strip().lstrip('v')
                print(f"On exact tag, using version: {current_version}")
            else:
                # Not on exact tag, use git describe
                result = subprocess.run(
                    ['git', 'describe', '--tags', '--match', 'v*'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    # Convert git describe output (e.g., v0.1.0-5-gabcdef) to PEP 440
                    git_version = result.stdout.strip().lstrip('v')
                    # Extract base version (before the commit count)
                    if '-' in git_version:
                        current_version = git_version.split('-')[0]
                    else:
                        current_version = git_version
                    print(f"From git describe, using version: {current_version}")
                else:
                    # Fallback version from setuptools_scm config
                    fallback = data.get('tool', {}).get('setuptools_scm', {}).get('fallback_version', '0.0.1')
                    current_version = fallback
                    print(f"Using fallback version: {current_version}")

        # Remove 'version' from dynamic list if present
        if 'dynamic' in data['project'] and 'version' in data['project']['dynamic']:
            data['project']['dynamic'].remove('version')
            if not data['project']['dynamic']:
                del data['project']['dynamic']

        data['project']['version'] = f"{current_version}+{cuda_tag}"

    with open('pyproject.toml', 'w') as f:
        tomlkit.dump(data, f)

    print(f"Added {len(deps)} CUDA dependencies to pyproject.toml")
    if cuda_tag:
        print(f"Set version to {data['project']['version']}")

if __name__ == '__main__':
    deps = sys.argv[1] if len(sys.argv) > 1 else ''
    cuda_tag = sys.argv[2] if len(sys.argv) > 2 else None
    add_dependencies(deps, cuda_tag)
