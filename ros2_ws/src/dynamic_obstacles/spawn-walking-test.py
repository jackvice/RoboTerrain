#!/usr/bin/env python3

import subprocess

def test_spawn():
    # Actor SDF with proper animation URL
    test_sdf = '''<?xml version="1.0" ?>
<sdf version="1.6">
  <actor name="test_actor">
    <pose>0 0 1.0 0 0 0</pose>
    <skin>
      <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae</filename>
      <scale>1.0</scale>
    </skin>
    <animation name="walking">
      <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae</filename>
      <scale>1.0</scale>
      <interpolate_x>true</interpolate_x>
    </animation>
  </actor>
</sdf>'''

    # Save to temp file
    with open('/tmp/test_actor.sdf', 'w') as f:
        f.write(test_sdf)
        print("SDF file written to /tmp/test_actor.sdf")

    # Print file contents for verification
    print("\nSDF file contents:")
    print(test_sdf)

    # Spawn command
    command = [
        'ign', 'service', 
        '-s', '/world/default/create',
        '--reqtype', 'ignition.msgs.EntityFactory',
        '--reptype', 'ignition.msgs.Boolean',
        '--timeout', '1000',
        '--req', 'sdf_filename: "/tmp/test_actor.sdf"'
    ]

    print("\nExecuting command:", ' '.join(command))
    
    # Run command
    result = subprocess.run(command,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True)

    print("\nCommand output:")
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    print("Return code:", result.returncode)

if __name__ == '__main__':
    test_spawn()
