#pragma once
#include <vector>
#include <cmath>

// Paramètres
int m = 1;

// Constantes
const double pi = acos(-1.0);

// Classes

class Noeud
{
public:
    int i = 0;
    int j = 0;
    int numgb(int N);
};

class Triangle
{
public:
    std::vector<double> x; // coordonnées x
    std::vector<double> y; // coordonnées y

    Triangle(double x0, double x1, double x2, double y0, double y1, double y2) {
        x = { x0, x1, x2 };
        y = { y0, y1, y2 };
    }
};

class ETA {
public:
    virtual double eta(double x, double y);               // fonction eta
    virtual double integ_eta(const Triangle& T);      // intégrale de eta sur un triangle

};

class VhElement {
public:
    std::vector<double> V;
    double L2norm;          // norme L2
    double gradL2norm;      // norme L2 du gradient
    double Linfininorm;     // norme L infini aussi surprenant que ça puisse paraitre

    VhElement(int G) : V(G, 0.0), L2norm(0.0), gradL2norm(0.0), Linfininorm(0.0) {}
    void normL2(int N, int M, double a, double b); // Fonction pour calculer la norme L2 de V
    void normL2Grad(int N, int M, double a, double b); // Fonction pour calculer la norme L2 du gradient de V
    void normLinfini(); // Fonction pour calculer la norme L2 du gradient de V

};

class RHSF {
private:
    double eta0;

public:
    RHSF(double eta0) : eta0(eta0) {}  // constructeur

    double f(double x, double y) {
        double C = 1.0 / (2.0 * m * m * pi * pi + 1.0);
        double eta = 1.0 - (eta0 * (x * x + y * y)) / 4.0; 
        double term1 = C * (2.0 * m * m * pi * pi * eta + 1.0)
            * cos(m * pi * x) * cos(m * pi * y);
        double term2 = eta0 * C * m * pi / 2.0
            * (x * sin(m * pi * x) * cos(m * pi * y)
                + y * cos(m * pi * x) * sin(m * pi * y));
        return term1 - term2;
    }
};

class SolExa {
public:
    double u_p(double x, double y) {
        return (cos(m*pi*x)*cos(m*y*pi))/(1+2*(m*pi)* (m * pi));   // solution exacte du problème trouvée en partie 1
    }
};

class ETAEta0 : public ETA {
private:
    double eta0;  // stocké comme attribut

public:
    ETAEta0(double eta0) : eta0(eta0) {}  // constructeur

    double eta(double x, double y) {
        return 1.0 - (eta0 / 4.0) * (x * x + y * y);
    }

    double integ_eta(const Triangle& T) {
        double x1 = (T.x[0] + T.x[1]) / 2, y1 = (T.y[0] + T.y[1]) / 2;
        double x2 = (T.x[1] + T.x[2]) / 2, y2 = (T.y[1] + T.y[2]) / 2;
        double x3 = (T.x[0] + T.x[2]) / 2, y3 = (T.y[0] + T.y[2]) / 2;
        double detBT = fabs((T.x[1] - T.x[0]) * (T.y[2] - T.y[0])
            - (T.x[2] - T.x[0]) * (T.y[1] - T.y[0]));
        return (detBT / 2.0) * (eta(x1, y1) + eta(x2, y2) + eta(x3, y3)) / 3.0;
    }
};
