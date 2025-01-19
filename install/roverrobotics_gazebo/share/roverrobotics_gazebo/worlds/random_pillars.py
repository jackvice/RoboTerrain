import random

def generate_pillar_xml():
    pillars_xml = ""
    
    for i in range(16):
        x = random.uniform(-8.0, 8.0)
        y = random.uniform(-8.0, 8.0)
        
        pillar = f"""        <model name='pillar_{i+1}'>
            <pose>{x} {y} 1 0 0 0</pose>
            <static>true</static>
            <link name='pillar_{i+1}_link'>
                <collision name='pillar_{i+1}_collision'>
                    <geometry>
                        <cylinder>
                            <radius>0.125</radius>
                            <length>2.0</length>
                        </cylinder>
                    </geometry>
                </collision>
                <visual name='pillar_{i+1}_visual'>
                    <geometry>
                        <cylinder>
                            <radius>0.125</radius>
                            <length>2.0</length>
                        </cylinder>
                    </geometry>
                    <material>
                        <ambient>0 0 0 1</ambient>
                        <diffuse>0 0 0 1</diffuse>
                        <specular>0 0 0 1</specular>
                    </material>
                </visual>
            </link>
        </model>
"""
        pillars_xml += pillar
    
    return pillars_xml

if __name__ == "__main__":
    # Generate and print the XML
    print(generate_pillar_xml())
