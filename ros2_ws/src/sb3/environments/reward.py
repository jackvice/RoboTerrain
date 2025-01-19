def task_reward():
        """
        Reward function that accounts for robot dynamics and gradual acceleration
        """
        # Constants
        final_reward_multiplier = 1.5
        collision_threshold = 0.3
        collision_penalty = -0.5
        success_distance = 0.75
        distance_delta_scale = 0.3
        heading_tolerance = math.pi/4  # 45 degrees

        # Get current state info
        distance_heading_info = self.get_target_info()
        current_distance = distance_heading_info[0]
        heading_diff = distance_heading_info[1]

        # Initialize previous distance if needed
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return 0.0
        
        # Check for goal achievement
        if current_distance < success_distance:
            self.update_target_pos()
            return self.goal_reward

        # Check for collisions
        min_distance = np.min(self.lidar_data[np.isfinite(self.lidar_data)])
        if min_distance < collision_threshold:
            print('Collision!')
            return collision_penalty
        
        # Calculate distance change (positive means got closer, negative means got further)
        distance_delta = self.previous_distance - current_distance

        # Calculate reward components
        distance_reward = 0.0
        heading_reward = 0.0
        

        # Heading component - reward facing towards target even if not moving much
        # This helps during acceleration phases
        heading_alignment = 1.0 - (abs(heading_diff) / math.pi)  # 1.0 when perfect, 0.0 when opposite
        heading_reward = 0.01 * heading_alignment  # 0.01 per step when perfect (30.0 over 3000 steps)
        # Heading component with new alignment calculation
        # Convert heading difference to range [-π, π]
        heading_diff = math.atan2(math.sin(heading_diff), math.cos(heading_diff))
        abs_heading_diff = abs(heading_diff)
        # Calculate heading alignment:
        # - When abs_heading_diff = 0 (perfect): heading_alignment = 1
        # - When abs_heading_diff = π/2 (90 degrees): heading_alignment = 0
        # - When abs_heading_diff = π (180 degrees): heading_alignment = -1
        if abs_heading_diff <= math.pi/2:
            # From 0 to 90 degrees: scale from 1 to 0
            heading_alignment = 1.0 - (2 * abs_heading_diff / math.pi)
        else:
            # From 90 to 180 degrees: scale from 0 to -1
            heading_alignment = -2 * (abs_heading_diff - math.pi/2) / math.pi
        heading_reward = 0.01 * heading_alignment  # 0.01 per step when perfect (30.0 over 3000 steps)

        # Distance component - reward any progress towards goal
        if abs(distance_delta) > 0.001 and heading_reward > 0.0:  # Only reward meaningful movement with good heading
            distance_reward = distance_delta * distance_delta_scale
        else:
            distance_reward = -0.001
            
        # Combine rewards
        reward = ((distance_reward + heading_reward) * final_reward_multiplier) + (self.current_linear_velocity * 0.0025)

        # Debug logging
        if self.total_steps % 1000 == 0:
            print(f"Distance: {current_distance:.3f}, Previous Distance: {self.previous_distance:.3f}, "
                  f"distance_delta: {distance_delta:.3f}, Heading diff: {math.degrees(heading_diff):.1f}°, "
                  f"Speed: {self.last_speed:.3f}, Current vel: {self.current_linear_velocity:.3f}, "
                  f"Distance reward: {distance_reward:.3f}, Heading reward: {heading_reward:.3f}, "
                  f"Total reward: {reward:.3f}")

        self.previous_distance = current_distance
        return reward
