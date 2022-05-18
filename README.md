## First Homework: Molecular docking application 

Create a molecular docking application. SPACE always results in two molecules, with a Coulomb force between the atoms of the two molecules, which moves and rotates the molecules. The atoms of the molecules are subjected to an interfacial drag proportional to their velocity. A molecule is a rigid structure of atoms with a random graph topology. The number of atoms is a random number between 2 and 8. The mass of the constituent atoms is a random positive integer multiple of the mass of the hydrogen atom and the charge of the electron. The total charge for each molecule is zero. The molecules move in 2D Euclidean space, the atoms here are circular, the edges of the graph within the atom are white and are sections in Euclidean geometry. Positively charged atoms are red and negatively charged atoms are blue, with intensity proportional to charge. Our microscope maps the Euclidean plane onto the hyperbolic plane, preserving the x, y coordinates, and then displays it using the Beltrami-Poincaré mapping on a circle of maximum radius that can be drawn on a 600x600 resolution screen. The s,d,x,e keys can be used to shift the Euclidean virtual world left, right, down and up by 0.1 units. The time step can be 0.01 sec regardless of the drawing speed. 

<img src="https://raw.githubusercontent.com/zsdrahos/opengl_graphics/main/images/1.png" alt="drawing" width="300"/>  <img src="https://raw.githubusercontent.com/zsdrahos/opengl_graphics/main/images/2.png" alt="drawing" width="300"/>

## Second Homework: Luxo-Grandpa ray-tracing 

Luxo Junior has become Luxo Grandpa (LG), so we are dedicating a new programme to it. LG stands on a plane, its structure (from bottom to top) is: cylinder base, ball joint1, cylinder rod1, ball joint2, cylinder rod2, ball joint3, paraboloid, with a point source sitting at the focal point. The ball joints rotate continuously along axes other than the coordinate axes. The actors are illuminated by another point light source and ambient light. The camera rotates around LG. The actors are of diffuse-specular type of lumpy material

<img src="https://raw.githubusercontent.com/zsdrahos/opengl_graphics/main/images/3.png" alt="drawing" width="300"/>  

## Third Homework: Luxo-Grandpa 3D incremental image synthesis

Solve the second homework problem using incremental image synthesis. The rotation of the lamp's joints is continuous, the camera circles around the scene and the lamp is always visible. The surfaces are of the lumpy type. Shadow implementation is not required.

<img src="https://raw.githubusercontent.com/zsdrahos/opengl_graphics/main/images/4.png" alt="drawing" width="300"/>   <img src="https://raw.githubusercontent.com/zsdrahos/opengl_graphics/main/images/5.png" alt="drawing" width="300"/>  
