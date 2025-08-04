<?xml version="1.0" ?>
<sdf version='1.9'>
  <world name='inspect'>
    
    <!-- Bullet Physics System Plugin -->
    <plugin
      filename="ignition-gazebo-physics-system"
      name="ignition::gazebo::systems::Physics">
      <engine>
        <filename>ignition-physics-bullet-plugin</filename>
      </engine>
      <max_step_size>0.005</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>200</real_time_update_rate>
    </plugin>
    
    <!-- Scene Broadcaster -->
    <plugin filename="ignition-gazebo-scene-broadcaster-system"
            name="ignition::gazebo::systems::SceneBroadcaster">
    </plugin>
    
    <!-- User Commands (for GUI interaction) -->
    <plugin filename="ignition-gazebo-user-commands-system"
            name="ignition::gazebo::systems::UserCommands">
    </plugin>
    
    <!-- Lighting -->
    <light name='sun' type='directional'>
      <cast_shadows>1</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 -0.5 -1</direction>
    </light>
    
    <!-- World Properties -->
    <gravity>0 0 -9.8</gravity>
    
    <!-- Scene Configuration -->
    <scene>
      <ambient>0.8 0.8 0.8 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>1</shadows>
    </scene>
    
    <!-- GUI Configuration -->
    <gui fullscreen='0'>
      <camera name='user_camera'>
        <pose>87.2825 25.0441 36.2306 0 0.363643 -3.06639</pose>
        <view_controller>orbit</view_controller>
        <projection_type>perspective</projection_type>
      </camera>
    </gui>
    
    <!-- Inspection World Terrain Model -->
    <model name='inspection_geom'>
      <static>true</static>
      <pose>0 0 0 0 0 0</pose>
      
      <link name='inspection_world_link'>
        <pose>0 0 0 0 0 0</pose>
        
        <!-- Collision with Bullet-optimized settings -->
        <collision name='inspection_world_link_collision'>
          <geometry>
            <mesh>
              <scale>1 1 1</scale>
              <uri>model://roverrobotics_gazebo/meshes/inspection_world.dae</uri>
            </mesh>
          </geometry>
          <surface>
            <friction>
              <bullet>
                <friction>2.0</friction>
                <friction2>2.0</friction2>
                <rolling_friction>0.1</rolling_friction>
              </bullet>
            </friction>
            <contact>
              <bullet>
                <kp>1000000</kp>
                <kd>1</kd>
              </bullet>
            </contact>
          </surface>
        </collision>
        
        <!-- Visual -->
        <visual name='inspection_world_link_visual'>
          <geometry>
            <mesh>
              <scale>1 1 1</scale>
              <uri>model://roverrobotics_gazebo/meshes/inspection_world.dae</uri>
            </mesh>
          </geometry>
        </visual>
        
      </link>
    </model>
    
  </world>
</sdf>
