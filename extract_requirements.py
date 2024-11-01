import yaml

with open('environment.yml') as f:
    env = yaml.safe_load(f)

pip_deps = []
for dep in env['dependencies']:
    if isinstance(dep, dict) and 'pip' in dep:
        pip_deps.extend(dep['pip'])

with open('requirements.txt', 'w') as f:
    for dep in pip_deps:
        f.write(f"{dep}\n")