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
#include "framework.h"
const char* const vertexSource = R"(
	#version 330				
	precision highp float;		
 
	uniform mat4 MVP;			
	layout(location = 0) in vec3 vp;
	
	void main() {
		float w = vp.z;
		gl_Position = vec4(vp.x/w, vp.y/w, 0, 1) * MVP;
	}
)";
 
const char* const fragmentSource = R"(
	#version 330			
	precision highp float;	
	
	uniform vec3 color;		
	out vec4 outColor;		
 
	void main() {
		outColor = vec4(color, 1);	
	}
)";
 
GPUProgram gpuProgram;
 
 
static float distance(vec3 p1, vec3 p2)
{
	double x = ((p1.x * p2.x) + (p1.y * p2.y) - (p1.z * p2.z));
	return log(-x + sqrt(pow(-x, 2) - 1));
}
 
vec3 trans(vec3 tra, vec3& to, vec3 cam = vec3(0.0, 0.0, 1.0))
{
	if (distance(cam, tra) < 0.000001) { return 0; }
	vec3 wend = 0.5 * (cam + tra) * (1 / (cosh(distance(cam, tra) / 2)));
	vec3 wstart = 0.5 * (1 / (sinh(distance(cam, tra)))) * ((2 * cam * sinh(distance(cam, tra) * 0.1)) + (2 * tra * sinh(distance(cam, tra) * 0.9)));
	vec3 q = 2 * wend * cosh(distance(to, wend)) - to;
	to = 2 * wstart * cosh(distance(q, wstart)) - q;
	to.z = sqrt(to.x * to.x + to.y * to.y + 1.0);
	return to;
}
 
 
static float random(float min, float max)
{
	return min + (rand()) / ((RAND_MAX / (max - min)));
}
 
 
class Atom
{
private:
	float radius;
	vec3 vel;
	vec3 pos;
	float weigth;
	float charge;
	int edges;
public:
	static const unsigned int vertNum = 30;
	vec3 vertices[vertNum];
	vec3 center;
	void setVel(vec3 v) { vel = v; }
	vec3 getVel() { return vel; }
	void setPos(vec3 v) { pos = v; }
	vec3 getPos() { return pos; }
	float getCharge() { return charge; }
	void setCharge(float c) { charge = c; }
	float getWeight() { return weigth; }
 
	Atom(vec3 c = vec3(0.0f, 0.00f, 1.0f))
	{
		float r_weight = random(0.03f, 0.08f);
		float r_charge = random(-1.0f, 1.0f);
 
		center = c;
		charge = r_charge;
		weigth = r_weight;
		radius = weigth;
 
		for (int i = 0; i < vertNum; ++i)
		{
			float phi = (2.0f * i * M_PI / vertNum);
			float x = cos(phi) * radius, y = sin(phi) * radius;
			vertices[i] = vec3(x, y, sqrt(x * x + y * y + 1.0f));
			trans(center, vertices[i]);
		}
		vec3 v = vec3(0, 0, 1);
		trans(center, v);
		center = v;
	}
};
 
class Molecule {
private:
	unsigned int vao, vbo;
	int nodeCount;
	float zerocharge = 0;
	std::vector<std::vector<int>> edges;
	std::vector<std::vector<vec3>> eCoordinate;
public:
	std::vector<Atom> graphNodes;
	int getNodeCount() { return nodeCount; }
	bool connected(int i1, int i2) {
		for (int j = 0; j < nodeCount; ++j) {
			if ((edges[j][0] == i1 && edges[j][1] == i2) || (edges[j][1] == i1 && edges[j][0] == i2))
			{
				return true;
			}
		}
		return false;
	}
 
	Molecule(int n)
	{
		srand(66);
		nodeCount = n;
		edges.resize(nodeCount, std::vector<int>(2, 0));
		eCoordinate.resize(nodeCount, std::vector<vec3>(2, 0));
 
		std::vector<int> ran(nodeCount - 2);
		std::vector<int> deg(nodeCount, 0);
 
 
		for (int i = 0; i < Atom::vertNum; ++i)
		{
			float phi = (2.0f * i * M_PI / Atom::vertNum);
			float x = (cos(phi) * 0.25f) + 0.25f, y = (sin(phi) * 0.25f) + 0.25;
		}
 
		//Prufer sequence from wikipedia
		for (int i = 0; i < nodeCount - 2; i++)
		{
			ran[i] = (rand() % (nodeCount));
		}
 
		for (int i = 0; i < nodeCount - 2; i++)
		{
			int first = ran[i] - 1;
			deg[first] += 1;
		}
 
		for (int i = 0; i < nodeCount - 2; i++)
		{
			for (int j = 0; j < nodeCount; j++)
			{
				int first = ran[i] - 1;
				if (deg[j] == 0)
				{
					deg[j] = -1;
					edges[j][0] = j + 1;
					edges[j][1] = ran[i];
					deg[first]--;
					break;
				}
			}
		}
 
		int x = 0;
		for (int i = 0; i < nodeCount; i++)
		{
			if (connected(deg[i], x))
			{
				edges[i][0] = i + 1;
				x++;
			}
			else if (connected(deg[i], x))
				edges[i][1] = i + 1;
		}
 
		for (int i = 0; i < nodeCount; ++i)
		{
			float x = random(-1.0f, 1.0f), y = random(-1.0f, 1.0f);
			float w = x * x + y * y + 1;
			graphNodes.push_back(Atom(vec3(x, y, sqrt(w))));
			zerocharge += graphNodes[i].getCharge();
		}
		if (zerocharge >= 0)
		{
			float tmp;
			graphNodes[nodeCount].setCharge(zerocharge * -1);
			tmp = zerocharge;
			zerocharge += graphNodes[nodeCount].getCharge();
			graphNodes[nodeCount].setCharge((tmp * -1) / 10);
		}
		else {
			float tmp;
			graphNodes[nodeCount].setCharge(zerocharge);
			tmp = zerocharge;
			zerocharge += graphNodes[nodeCount].getCharge();
			graphNodes[nodeCount].setCharge((tmp) / 10);
		}
 
		for (int i = 0; i < nodeCount; i++)
		{
			for (int k = 0; k < nodeCount; k++) {
				if (connected(i, k) && distance(graphNodes[i].center, graphNodes[k].center) > 2.0 && i != k)
				{
					float x = random(-1.0f, 1.0f), y = random(-1.0f, 1.0f);
					float w = x * x + y * y + 1;
					for (int j = 0; j < Atom::vertNum; ++j)
					{
						trans(vec3(x, y, sqrt(w)), graphNodes[i].vertices[j]);
					}
				}
			}
		}
	}
 
	void Init() {
		for (int i = 0; i < nodeCount; i++)
		{
			for (int j = 0; j < nodeCount; j++) {
				if (edges[j][0] == i)
				{
					vec3 p = graphNodes[i].center;
					p.z = sqrt(p.x * p.x + p.y * p.y + 1.0);
					eCoordinate[j][0] = p;
				}
				if (edges[j][1] == i)
				{
					vec3 p = graphNodes[i].center;
					p.z = sqrt(p.x * p.x + p.y * p.y + 1.0);
					eCoordinate[j][1] = p;
				}
			}
		}
 
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
 
	}
 
	void draw() {
 
		for (int i = 0; i < nodeCount; ++i)
		{
			vec3 drawEdges[2];
			drawEdges[0] = eCoordinate[i][0];
			drawEdges[1] = eCoordinate[i][1];
			glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * 2, &drawEdges[0], GL_DYNAMIC_DRAW);
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, 1.0f, 1.0f, 1.0f);
			glDrawArrays(GL_LINES, 0, 2);
		}
 
		for (int i = 0; i < nodeCount; ++i)
		{
			vec3 drawVertices[Atom::vertNum];
			for (int j = 0; j < Atom::vertNum; ++j) {
				drawVertices[j] = graphNodes[i].vertices[j];
			}
 
			glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * Atom::vertNum, &drawVertices[0], GL_DYNAMIC_DRAW);
 
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
 
			if (graphNodes[i].getCharge() <= 0)
			{
				glUniform3f(location, 0.0f, 0.0f, (graphNodes[i].getCharge() * -1));
			}
			else if (graphNodes[i].getCharge() >= 0) {
				glUniform3f(location, graphNodes[i].getCharge(), 0.0f, 0.0f);
			}
			glDrawArrays(GL_TRIANGLE_FAN, 0, Atom::vertNum);
		}
	}
};
 
std::vector<Molecule*> graphs;
float p = 0, q = 0;
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glPointSize(10.0f);
	glLineWidth(2.0f);
 
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}
void onDisplay() {
	glClearColor(0.6, 0.6, 0.6, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	mat4 MVPtransf = TranslateMatrix(vec3(p, q, 0.0));
	gpuProgram.setUniform(MVPtransf, "MVP");
	for (auto c : graphs)
	{
		c->Init();
		c->draw();
	}
	glutSwapBuffers();
}
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();
	if (key == ' ') {
		graphs.push_back(new Molecule(random(2, 8)));
		graphs.push_back(new Molecule(random(2, 8)));
		glutPostRedisplay();
	}
 
	switch (key) {
	case 's': p -= 0.1f; break;
	case 'd': p += 0.1f; break;
	case 'x': q -= 0.1f; break;
	case 'e': q += 0.1f; break;
	}
	glutPostRedisplay();
}
void onKeyboardUp(unsigned char key, int pX, int pY) {
 
}
 
void onMouseMotion(int pX, int pY) {
 
}
 
void onMouse(int button, int state, int pX, int pY) {
 
}
 
float dt = 0.01f;
float coulombF;
float ckonst = 0.8988;
float wSum;
float theta;
float friction_cd = 0.47;
float specific_mass = 1;
void onIdle() {
 
	for (auto c : graphs)
	{
		for (int i = 0; i < c->getNodeCount(); ++i)
		{
			vec3 v;
			float d;
			for (int j = 0; j < c->getNodeCount(); ++j)
			{
				d = distance(c->graphNodes[i].center, c->graphNodes[j].center);
				if (i != j) {
 
					coulombF = ckonst * (c->graphNodes[i].getCharge()) * (c->graphNodes[j].getCharge()) / pow(d, 2.0);
					v = v + normalize((c->graphNodes[i].center - (c->graphNodes[j].center))) * coulombF;
					normalize(v);
 
					wSum = d * coulombF;
					theta = c->graphNodes[i].getWeight() * d * d;
				}
			}
 
			float w = (wSum / theta * dt);
			vec3 acc = w * dt;
			vec3 vell = c->graphNodes[i].getVel() + acc * dt + ((-1 * c->graphNodes[i].center + v) / c->graphNodes[i].getWeight()) * dt;
 
			vell = vell - vell * friction_cd * specific_mass;
			vell.z = sqrt(vell.x * vell.x + vell.y * vell.y + 1);
			c->graphNodes[i].setVel(vell);
			c->graphNodes[i].setPos(vell);
		}
 
		for (int i = 0; i < c->getNodeCount(); ++i)
		{
			trans(c->graphNodes[i].getPos(), c->graphNodes[i].center);
			for (int j = 0; j < Atom::vertNum; ++j)
			{
				trans(c->graphNodes[i].getPos(), c->graphNodes[i].vertices[j]);
			}
		}
	}
 
	glutPostRedisplay();
 
}