//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Drahos Zsolt
// Neptun : UCZFU3
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
// reszt vettem a hivatalos konzultacion!
#include "framework.h"
const float epsilon = 0.0001f;

vec4 qmul(vec4 q1, vec4 q2) {
	vec3 d1(q1.x, q1.y, q1.z), d2(q2.x, q2.y, q2.z);
	vec3 imag = d2 * q1.w + d1 * q2.w + cross(d1, d2);
	return vec4(imag.x, imag.y, imag.z, q1.w * q2.w - dot(d1, d2));
}

vec4 quaternion(float ang, vec3 axis) {
	vec3 d = normalize(axis) * sinf(ang / 2);
	return vec4(d.x, d.y, d.z, cosf(ang / 2));
}

vec4 qinv(vec4 q) {
	return vec4(-q.x, -q.y, -q.z, q.w);
}

vec3 Rotate(vec3 u, vec4 q)
{
	vec4 qinv(-q.x, -q.y, -q.z, q.w);
	vec4 qr = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), qinv);
	return vec3(qr.x, qr.y, qr.z);
}

float frame = 15;
vec4 q = quaternion(frame, normalize(vec3(0, 2, 1)));

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

vec3 operator/(vec3 num, vec3 denom)
{
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
	
public:
	float type;
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;
	vec3 normal;

	Sphere(const vec3& _center, float _radius, Material* _material, float place, vec3 _normal) {
		center = _center;
		radius = _radius;
		material = _material;
		type = place;
		normal = _normal;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		normal = normalize((hit.position - center) * (1.0f / radius));
		if (type == 3) {
			hit.normal = Rotate(normal, (q));
		}
		else if (type == 33)
		{
			hit.normal = Rotate(Rotate(normal, (q)), qinv(q));
		}
		else {
			hit.normal = normal;
		}
		
		hit.material = material;
		return hit;
	}
};

struct Paraboloid : public Intersectable {
	mat4 Q; 
	float zmin, zmax;
	vec3 translation;
	vec3 normal;

	Paraboloid(float _zmin, float _zmax, vec3 _translation, Material* _material, float place, vec3 _normal)
	{
		Q = mat4(40, 0, 0, 0,
				0, 40, 0, 0,
				0, 0, 0, 1,
				0, 0, 1, 0);
		zmin = _zmin; zmax = _zmax;
		translation = _translation;
		material = _material;
		type = place;
		normal = _normal;
	}
	vec3 gradf(vec3 r) {
		vec4 g = vec4(r.x, r.y, r.z, 1) * Q * 0.5;
		return vec3(g.x, g.y, g.z);
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 start = ray.start - translation;
		vec4 S(start.x, start.y, start.z, 1), D(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		float a = dot(D * Q, D);
		float b = dot(S * Q, D) * 2;
		float c = dot(S * Q, S);

		float discr = b * b - 4.0f * a * c;

		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);

		float t1 = (-b + sqrt_discr) / 2.0f / a;
		vec3 p1 = ray.start + ray.dir * t1;
		if (p1.z < zmin || p1.z > zmax) t1 = -1;

		float t2 = (-b - sqrt_discr) / 2.0f / a;
		vec3 p2 = ray.start + ray.dir * t2;
		if (p2.z < zmin || p2.z > zmax) t2 = -1;


		if (t1 <= 0 && t2 <= 0) return hit;

		if (t1 <= 0) hit.t = t2;
		else if (t2 <= 0) hit.t = t1;
		else if (t2 < t1) hit.t = t2;
		else hit.t = t1;
		
		normal = normalize(gradf(start + ray.dir * hit.t));
		hit.normal = normalize(Rotate(Rotate(normal,(q)),qinv(q)));
		
		hit.position = hit.position + translation;
		hit.material = material;
		return hit;
	}
};

struct Henger : public Intersectable {
	vec3 center;
	float r;
	float h;
	vec3 normal;
	Henger(vec3 c, float r_, float h_, Material* _material, float place, vec3 _normal) {
		center = c;
		r = r_;
		h = h_;
		material = _material;
		normal = _normal;
		type = place;
	}
	vec3 gradf(vec3 r) {
		vec4 g = vec4(r.x, r.y, r.z, 1) * ScaleMatrix(vec3(-9,-9,0)) * 0.5;
		return vec3(g.x, g.y, g.z);
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		float a = ray.dir.x * ray.dir.x + ray.dir.z * ray.dir.z;
		float b = 2 * ((ray.start.x - center.x) * ray.dir.x + (ray.start.z - center.z) * ray.dir.z);
		float c = (ray.start.x - center.x) * (ray.start.x - center.x) + (ray.start.z - center.z) * (ray.start.z - center.z) - (r * r);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		hit.t = (t2 > 0) ? t2 : t1;
		if (hit.t != 0) {
			
			if ((ray.start.y + ray.dir.y * t1) > center.y + h)
			{
				hit.t = 0;
			}
			else if ((ray.start.y + ray.dir.y * t2) < center.y)
			{
				hit.t = 0;
			}
			hit.position = ray.start + ray.dir * hit.t;
			normal = ((ray.start + ray.dir * hit.t)-center*hit.t);
			
			if (type == 2)
				hit.normal = normalize(Rotate(normal, q));
			else if (type == 22)
				hit.normal = normalize(Rotate(Rotate(normal, q), qinv(q)));
			else
				hit.normal = normalize(normal);
			hit.normal.y = 0.0;
			if (ray.dir.y < epsilon) {
				vec3 dist = vec3(center.x, center.y + h, center.z) - ray.start;
				float t_circle = dist.y / ray.dir.y;
				if (dot(ray.dir * t_circle - dist, ray.dir * t_circle - dist) <= r * r)
				{
					hit.t = t_circle;
					hit.position = (ray.start + ray.dir * hit.t);
					normal = vec3(0,1,0);
					
					if (type == 2)
						hit.normal = normalize(Rotate(normal, q));
					else if (type == 22)
						hit.normal = normalize(Rotate(Rotate(normal, q), qinv(q)));
					else
						hit.normal = normalize(normal);

				}
			}
		}
		hit.material = material;
		return hit;
	}
	vec3 N() {
		return normal;
	}
};

struct Plane : public Intersectable {
	vec3 point;
	Plane(vec3 p, Material* _material) {
		point = p;
		material = _material;
		type = 0;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		
		hit.normal = ((vec3(0,1, 0)));
		hit.t = dot(point - ray.start, hit.normal) / dot(ray.dir, hit.normal);
		hit.material = material;
		hit.position = ray.start + ray.dir * hit.t;
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;

public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		fov = _fov;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		float windowSize = length(w) * tanf(fov / 2);
		right = normalize(cross(vup, w)) * windowSize;
		up = normalize(cross(w, right)) * windowSize;
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void Animate(float dt) {
		
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x, eye.y, -(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, vec3(0,1,0), fov);
	}

};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

struct PointLight {
	vec3 Le;
	vec3 pos;
	PointLight(vec3 _pos, vec3 _Le) {
		Le = _Le;
		pos = (_pos);
	}
};


class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	std::vector<PointLight*> pointlights;
	Camera camera;
	vec3 La;
public:
	void build() {
		float fov = 45 * (float)M_PI / 180;
		vec3 eye = vec3(0, 2, 4), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.06f, 0.06f, 0.06f);
		vec3 lightDirection(1, 1, 1), Le(0.7, 0.7, 0.7);
		lights.push_back(new Light(lightDirection, Le));
		
		vec3 _;

		vec3 kd(0.1f, 0.1f, 0.3f), ks(0.1, 0.1, 0.1);
		Material* material = new Material(kd, ks, 50);
		Material* material2 = new Material(vec3(0.4f, 0.2f, 0.05f), vec3(0.9, 0.9, 0.9), 50);
		Material* materialBulb = new Material(vec3(0.9f, 0.9f, 0.9f), vec3(10, 10, 10), 1);
		
		objects.push_back(new Plane(vec3(0, -0.1, 0), material2));
		objects.push_back(new Henger(vec3(0, -0.1, 0), 0.2f, 0.08f, material, 0, _));
		
		objects.push_back(new Sphere(vec3(0, 0, 0), 0.05f, material, 1, _));
		objects.push_back(new Henger(vec3(0, 0, 0), 0.02f, 0.35f, material, 2, _));

		objects.push_back(new Sphere(Rotate(vec3(0, 0.35, 0), (q)), 0.05f, material, 3, _));
		objects.push_back(new Henger(Rotate(vec3(0, 0.35, 0), (q)), 0.02f, 0.4f, material, 22, _));

		vec3 r_rotate = Rotate(vec3(0, 0.35, 0), (q));
		r_rotate.y += 0.4;
		vec3 rr_rotate = Rotate(r_rotate, (q));
		
		objects.push_back(new Sphere(Rotate(r_rotate, q), 0.05f, material, 33, _));
		objects.push_back(new Paraboloid(-0.1, 1, rr_rotate, material, 0, _));
		
		vec3 lpos = vec3(1.9, 1, -1);
		vec3 Le2(0.015, 0.015, 0.015);
		pointlights.push_back(new PointLight(lpos, Le2));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			
			Hit hit;
			if (object->type == 0 || object->type == 2 || object->type == 22) {
				ray.dir = ray.dir;
				ray.start = ray.start;
				hit = object->intersect(ray);
			}
			else {
				ray.dir = Rotate(ray.dir, q);
				ray.start = Rotate(ray.start, q); 
				hit = object->intersect(ray);
			}

			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal;
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		for (Intersectable* object : objects) 
			if (object->type == 0 || object->type == 2 || object->type == 22) {
				ray.dir = ray.dir;
				ray.start = ray.start;
				if (object->intersect(ray).t > 0)
					return true;
			}
			else {
				ray.dir = Rotate(ray.dir, q);
				ray.start = Rotate(ray.start, q);
				if (object->intersect(ray).t > 0)
					return true;
			}
			
		return false;
	}


	vec3 trace(Ray ray) {

		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance;


		for (Light* light : lights) {
			outRadiance = hit.material->ka * La;
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, normalize(light->direction));
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
			
		}


		for (PointLight* plight : pointlights) {

			vec3 shadedPoint = hit.position + hit.normal * epsilon;
			vec3 toLight = normalize(plight->pos - shadedPoint);
			float distToLight = length(toLight);
			toLight = toLight / distToLight;
			float t;
			float cosTheta = (dot(toLight, hit.normal));
			if (hit.t > 0.0) {
				float lightIntensity = 500;
				outRadiance = outRadiance + plight->Le * hit.material->kd * cosTheta / pow(distToLight, 2.0) * lightIntensity;
			}
		}
	

		return outRadiance;
	}
	void Animate(float dt) { camera.Animate(dt); }
};

GPUProgram gpuProgram;
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			
	out vec4 fragmentColor;		

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image) : texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);


		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		
		
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	fullScreenTexturedQuad->Draw();
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	glutSwapBuffers();
	
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	frame += 0.1;
	scene.Animate(0.1f);
	q = quaternion(frame, normalize(vec3(0, 2, 1)));
	glutPostRedisplay();
}