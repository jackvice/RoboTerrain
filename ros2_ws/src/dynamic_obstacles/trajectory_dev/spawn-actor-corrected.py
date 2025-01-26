#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SpawnEntity
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future
import os

class ActorSpawner(Node):
    def __init__(self):
        super().__init__('actor_spawner')
        
        # Use ReentrantCallbackGroup for async operations
        self.callback_group = ReentrantCallbackGroup()
        
        # Create spawn client
        self.spawn_client = self.create_client(
            SpawnEntity, 
            '/world/default/create',
            callback_group=self.callback_group
        )
        
        # Wait for spawn service
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for spawn service...')

    def load_trajectory_file(self, filepath: str) -> str:
        """Load trajectory from file"""
        try:
            with open(filepath, 'r') as f:
                trajectory_content = f.read().strip()
                
            # Verify the content looks like a trajectory block
            if not (trajectory_content.startswith("<trajectory") and 
                   trajectory_content.endswith("</trajectory>")):
                raise ValueError("File does not contain a valid trajectory block")
                
            return trajectory_content
        except FileNotFoundError:
            self.get_logger().error(f'Trajectory file not found: {filepath}')
            raise
        except Exception as e:
            self.get_logger().error(f'Error reading trajectory file: {e}')
            raise

    async def spawn_actor(self, trajectory_sdf: str, name: str = "walking_actor", 
                         x: float = 0.0, y: float = 0.0, z: float = 1.0) -> bool:
        """Spawn an actor with the given trajectory"""
        actor_sdf = f'''<?xml version="1.0" ?>
<sdf version="1.6">
  <actor name="{name}">
    <pose>{x} {y} {z} 0 0 0</pose>
    <skin>
      <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae</filename>
      <scale>1.0</scale>
    </skin>
    <animation name="walking">
      <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae</filename>
      <scale>1.0</scale>
      <interpolate_x>true</interpolate_x>
    </animation>
    {trajectory_sdf}
  </actor>
</sdf>'''

        try:
            # Create spawn request
            req = SpawnEntity.Request()
            req.xml = actor_sdf
            req.name = name
            
            # Call spawn service
            self.get_logger().info('Spawning actor...')
            future = await self.spawn_client.call_async(req)
            
            if future.success:
                self.get_logger().info(f'Successfully spawned actor: {name}')
                return True
            else:
                self.get_logger().error(f'Failed to spawn actor: {future.status_message}')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error spawning actor: {e}')
            return False

async def main(args=None):
    rclpy.init(args=args)
    spawner = ActorSpawner()

    try:
        # Load trajectory from file
        trajectory_file = 'trajectory.sdf'
        trajectory_sdf = spawner.load_trajectory_file(trajectory_file)
        
        # Spawn actor
        success = await spawner.spawn_actor(trajectory_sdf)
        
        if not success:
            spawner.get_logger().error('Failed to spawn actor')
            
    except Exception as e:
        spawner.get_logger().error(f'Error in main: {e}')
    finally:
        spawner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
