#include<iostream>
#include<vector>
#include<math.h>
#include "header.h"
#include<stdexcept>
#include <fstream>
#include <omp.h>




using namespace std;

// Old

vector<double> SoustrVect(vector<double> A, vector<double> B) {  // fonction pour soustraire deux vecteurs
	int n = A.size();
	vector<double> L(n, 0);
	for (int i = 0; i < n; i++)
		L[i] = A[i] - B[i];
	return L;
}

// Fin Old

double norme_euclidienne(const vector<double>& X) {
	double s = 0.0;
	for (int i = 0; i < X.size(); i++) {
		s += X[i] * X[i];
	}
	return sqrt(s);
}

vector<double> somme_vect(const vector<double>& X, const vector<double>& Y) {
	vector<double> Z(X.size());
	for (int i = 0; i < X.size(); i++) {
		Z[i] = X[i] + Y[i];
	}
	return Z;
}

int Noeud::numgb(int N) {
	return ((N + 1) * j + i);
}

vector<int> invnumgb(int N, int M, int s) {
	if (s > (N + 1) * (M + 1) - 1) {
		throw out_of_range("Numero du noeud hors du maillage");
	}
	vector<int> X(2, 0);
	X[0] = s % (N + 1); // Reste de la division euclidienne
	X[1] = s / (N + 1); // Quotient de la division euclidienne
	return X;
}

void visualiser_vecteur(vector<double> X) {
	for (int i = 0; i <= X.size() - 1; i++) {
		cout << X[i] << ",";
	}
	cout << endl;
}

void visualiser_vecteur_entier(vector<int> X) {
	for (int i = 0; i <= X.size() - 1; i++) {
		cout << X[i] << ",";
	}
	cout << endl;
}

void visualiser_mat_int(vector<vector<int>> M) {   // fonction pour visualiser une matrice
	int L = M.size();
	int l = M[0].size();
	for (int i = 0; i <= L - 1; i++) {
		for (int j = 0; j <= l - 1; j++) {
			cout << M[i][j] << ",";
		}
		cout << endl;
	}
}

vector<double> prod_par_scal(const vector<double>& X, double alpha) { // produit d'un 
	vector<double> Y(X.size());
	for (int i = 0; i < X.size(); i++) {
		Y[i] = alpha * X[i];
	}
	return Y;
}

double prod_scalaire(const vector<double>& X, const vector<double>& Y) {
	double s = 0.0;
	#pragma omp parallel for reduction(+:s)
	for (int i = 0; i < X.size(); i++) {
		s += X[i] * Y[i];
	}
	return s;
}

vector<double> Subdiv(double a, int N) {
	vector<double> X(N + 1, 0); // Création d'un vecteur vide
	double h = (2 * a) / N; // Partition du segment

	// Remplissage du vecteur X 
	for (int i = 0; i < N + 1; i++) {
		X[i] = i * h - a;
	}

	return(X);
}

vector<vector<int>> maillageTR(int N, int M) {
	int K = 2 * N * M;
	vector<vector<int>> TRG(K, vector<int>(3, 0));
	int k = 0;
	for (int j = 0; j < M; j++)
	{
		for (int i = 0; i < N; i++)
		{
			Noeud A, B, C, D;

			A.i = i;
			A.j = j;

			B.i = i + 1;
			B.j = j;

			C.i = i;
			C.j = j + 1;

			D.i = i + 1;
			D.j = j + 1;

			TRG[k][0] = A.numgb(N);
			TRG[k][1] = B.numgb(N);
			TRG[k][2] = D.numgb(N);
			k++;

			TRG[k][0] = A.numgb(N);
			TRG[k][1] = D.numgb(N);
			TRG[k][2] = C.numgb(N);
			k++;
		}
	}

	return TRG;
}

vector<vector<int>> maillageTR_bis(int N, int M) {
	int K = 2 * N * M;
	vector<vector<int>> TRG(K, vector<int>(3, 0));
	int k = 0;

	for (int j = 0; j < M; j++) {
		for (int i = 0; i < N; i++) {

			Noeud A, B, C, D;
			A.i = i;     A.j = j;
			B.i = i + 1; B.j = j;
			C.i = i;     C.j = j + 1;
			D.i = i + 1; D.j = j + 1;

			if ((i + j) % 2 == 0) {
				// Choix 1 — diagonale A-D
				TRG[k][0] = A.numgb(N);
				TRG[k][1] = B.numgb(N);
				TRG[k][2] = D.numgb(N);
				k++;
				TRG[k][0] = A.numgb(N);
				TRG[k][1] = D.numgb(N);
				TRG[k][2] = C.numgb(N);
				k++;
			}
			else {
				// Choix 2 — diagonale B-C
				TRG[k][0] = A.numgb(N);
				TRG[k][1] = B.numgb(N);
				TRG[k][2] = C.numgb(N);
				k++;
				TRG[k][0] = B.numgb(N);
				TRG[k][1] = D.numgb(N);
				TRG[k][2] = C.numgb(N);
				k++;
			}
		}
	}
	return TRG;
}

vector<vector<double>> CalcMatBT(const Triangle& T) {
	vector<vector<double>> BT(2, vector<double>(2, 0.0)); // On introduit une matrice vide
	BT[0][0] = T.x[1] - T.x[0]; //Matrice BT 
	BT[0][1] = T.x[2] - T.x[0];
	BT[1][0] = T.y[1] - T.y[0];
	BT[1][1] = T.y[2] - T.y[0];

	return BT;
}


double ETA::eta(double x, double y) {
	return x * x;    // Fonction arbitraire, de degré <= d=2 pour que la formule de quadrature soit cohérente
}

double ETA::integ_eta(const Triangle& T) {  // fonction pour calculer l'integrale de eta sur un triangle donné
	vector<vector<double>> BT = CalcMatBT(T); // on utilise CalcMatBT pour calculer BT pour le triangle T 
	double detBT = BT[0][0] * BT[1][1] - BT[0][1] * BT[1][0];
	double absdet = fabs(detBT); // determinant de BT 

	double integral = 0.0;
	//méthode de quadrature des points du milieu
	double x1 = 0.5 * T.x[0] + 0.5 * T.x[2];  // premier point du milieu
	double y1 = 0.5 * T.y[0] + 0.5 * T.y[2];
	integral += eta(x1, y1) / 3.0;

	double x2 = 0.5 * T.x[1] + 0.5 * T.x[2];  // deuxième point du milieu
	double y2 = 0.5 * T.y[1] + 0.5 * T.y[2];
	integral += eta(x2, y2) / 3.0;

	double x3 = 0.5 * T.x[0] + 0.5 * T.x[1];  // troisième point du milieu
	double y3 = 0.5 * T.y[0] + 0.5 * T.y[1];
	integral += eta(x3, y3) / 3.0;

	integral = integral * absdet / 2; //integral*aire du triangle

	return integral;
}

vector<double> integ_eta_triang(int N, int M, double a, double b, ETA& eta_obj) { // On parcours les triangles pour un domaine [-a,a]x[-b,b] donné, avec les aires de eta sur ces triangles
	vector<double> X = Subdiv(a, N);  // On subdivisie l'intervalle selectionné
	vector<double> Y = Subdiv(b, M);

	vector<vector<int>> maillage = maillageTR(N, M); // On maille l'intervalle
	int K = maillage.size();
	vector<double> ET(K, 0.0); // Vecteur ET qui stock l'aire de eta sur les triangles du maillage

	for (int k = 0; k < K; k++) {
		int s0 = maillage[k][0];   //On selectionne les points du maillage pour les sommets des triangles
		int s1 = maillage[k][1];
		int s2 = maillage[k][2];

		// On utilise invnumgb pour retrouver les coordonnées
		vector<int> ij0 = invnumgb(N, M, s0); //(x0,y0)
		vector<int> ij1 = invnumgb(N, M, s1); //(x1,y1)
		vector<int> ij2 = invnumgb(N, M, s2); //(x2,y2)

		Triangle T( // on introduit le triangle T, on le défini à l'aide des coordonnées qu'on vient de calculer
			X[ij0[0]], X[ij1[0]], X[ij2[0]],
			Y[ij0[1]], Y[ij1[1]], Y[ij2[1]]
		);

		ET[k] = eta_obj.integ_eta(T); // On applique la fonction integ_eta sur le triangle T qu'on vient de définir
	}  // La boucle répète, et calcule de cette manière tous les triangles successivement 

	return ET;
}

vector<double> DiffTerm(const vector<double>& xs, const vector<double>& ys, ETA& eta_obj)
{
	Triangle T(xs[0], xs[1], xs[2], ys[0], ys[1], ys[2]); // Triangle T défini par les coordonnées des vecteurs xs et ys

	vector<vector<double>> BT = CalcMatBT(T); // On calcul BT

	double val = eta_obj.integ_eta(T); // on calcul l'integrale de eta sur le triangle T

	double detBT = BT[0][0] * BT[1][1] - BT[0][1] * BT[1][0]; // determinant de BT

	vector<vector<double>> BTinvT(2, vector<double>(2)); // On calcul la transposée de B_T{-1}
	BTinvT[0][0] = BT[1][1] / detBT;
	BTinvT[0][1] = -BT[1][0] / detBT;
	BTinvT[1][0] = -BT[0][1] / detBT;
	BTinvT[1][1] = BT[0][0] / detBT;

	vector<vector<double>> grad_lambda = // on stock les gradients des lambda chapeau
	{
		{-1,-1},
		{ 1, 0},
		{ 0, 1}
	};

	vector<vector<double>> grad(3, vector<double>(2)); // vecteur pour stocker les (B_T{-1})^T*grad(lambda(i)^)

	for (int i = 0; i < 3; i++)
	{
		grad[i][0] = BTinvT[0][0] * grad_lambda[i][0] + BTinvT[0][1] * grad_lambda[i][1]; // On calcul le produit matriciel (B_T{-1})^T*grad(lambda(i)^)
		grad[i][1] = BTinvT[1][0] * grad_lambda[i][0] + BTinvT[1][1] * grad_lambda[i][1];
	}

	vector<double> A(6);

	int p = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = i; j < 3; j++)
		{
			double prod =
				grad[i][0] * grad[j][0] +
				grad[i][1] * grad[j][1]; //produit scalaire <(B_T{-1})^T*grad(lambda(i)^) , B_T{-1})^T*grad(lambda(j)^)>

			A[p] = prod * val;
			p += 1;
		}
	}

	return A;
}

vector<double> ReacTerm(const vector<double>& xs, const vector<double>& ys, double lambda)
{
	Triangle T(xs[0], xs[1], xs[2], ys[0], ys[1], ys[2]); // Triangle T défini par les coordonnées des vecteurs xs et ys

	vector<vector<double>> BT = CalcMatBT(T);  // On calcul BT

	double detBT = fabs(BT[0][0] * BT[1][1] - BT[0][1] * BT[1][0]);  // determinant de BT

	vector<double> A(6); // Vecte

	int p = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = i; j < 3; j++)
		{
			if (i == j)
				A[p++] = detBT * lambda * (1.0 / 12.0); // Calcul de l'integrale par rapport aux valeurs des integrales trouvées en question 27 (il suffit de multiplier par 1/12 si i=j, 1/24 sinon)
			else
				A[p++] = detBT * lambda * (1.0 / 24.0);
		}
	}

	return A;
}

int indices_sym(int i, int j) {
	if (i > j) { int x = i; i = j; j = x; }
	if (i == 0 && j == 0) return 0;
	if (i == 0 && j == 1) return 1;
	if (i == 0 && j == 2) return 2;
	if (i == 1 && j == 1) return 3;
	if (i == 1 && j == 2) return 4;
	if (i == 2 && j == 2) return 5;
	return -1;
}

vector<double> matvec(
	VhElement& vh_elem,
	const vector<double> &X, 
	const vector<double> &Y, 
	const vector<vector<int>> &maillage, 
	const vector<vector<double>> &precomp_diff,
	const vector<vector<double>>& precomp_reac,
	int N, int M, double lambda) {

	int G = (N + 1) * (M + 1); // dimension G de Vh
	vector<double> W(G, 0.0); // on crée un vecteur vide qui nous servira pour stocker les AV_k

	int K = maillage.size();

	for (int t = 0; t < K; t++) { // On parcours chaque triangle

		// sommets du triangle
		int s0 = maillage[t][0];
		int s1 = maillage[t][1];
		int s2 = maillage[t][2];

		// On calcul les intégrales des lambdas(i), ici ils coincident avec les w(i) puisque l'on est sur le triangle T et qu'on considere des sommets de ce meme triangles
		vector<double> ATDiff = precomp_diff[t];
		vector<double> ATReac = precomp_reac[t];


		int sommets[3] = { s0, s1, s2 }; // on stock les numéros des sommets

		// Boucle sur les sommets du triangle
		for (int i = 0; i < 3; i++) {
			int k = sommets[i];   // on a le numéro du noeud en question
			double res = 0.0;  // on initialise la somme

			// Boucle sur tous les sommets de T
			for (int j = 0; j < 3; j++) {
				int l = sommets[j]; // numéro du noeud correspondant au j-eme sommet de T
				int idx = indices_sym(i, j); // indice pour faire le lien avec les vecteurs de taille 6 définis dans DiffTerm et ReacTerm
				res += vh_elem.V[l] * (ATDiff[idx] + ATReac[idx]); // contribution du sommet j par rapport au sommet i
			}

			W[k] += res; // on ajoute la contribution totale de T au noeud 
		}
	}
	return W; // On renvoie W avec les valeurs de AV

}

vector<double> scdmembre(int N, int M, double a, double b, RHSF& rhsf)
{
	int G = (N + 1) * (M + 1); // dimension G de Vh
	vector<double> B(G, 0.0);  // on crée le vecteur pour stocker les B_k

	vector<double> X = Subdiv(a, N);
	vector<double> Y = Subdiv(b, M);

	vector<vector<int>> maillage = maillageTR(N, M);
	int K = maillage.size();

	for (int t = 0; t < K; t++) // on parcours chaque triangle, ici on va utiliser la formule de quadrature des points du milieu sur chaque triangle
	{
		// noeuds du maillage
		int s0 = maillage[t][0];
		int s1 = maillage[t][1];
		int s2 = maillage[t][2];
		// on recupere les coordonnées aux noeuds
		vector<int> ij0 = invnumgb(N, M, s0);
		vector<int> ij1 = invnumgb(N, M, s1);
		vector<int> ij2 = invnumgb(N, M, s2);

		Triangle T( // on défini T en fonction des coordonneés
			X[ij0[0]], X[ij1[0]], X[ij2[0]],
			Y[ij0[1]], Y[ij1[1]], Y[ij2[1]]
		);

		// Calcul de BT + determinant/aire
		double detBT = fabs((T.x[1] - T.x[0]) * (T.y[2] - T.y[0]) - (T.x[2] - T.x[0]) * (T.y[1] - T.y[0]));
		double aire = detBT / 2.0;
		// milieux des arêtes
		double xa = (T.x[0] + T.x[1]) / 2, ya = (T.y[0] + T.y[1]) / 2;
		double xb = (T.x[1] + T.x[2]) / 2, yb = (T.y[1] + T.y[2]) / 2;
		double xc = (T.x[2] + T.x[0]) / 2, yc = (T.y[2] + T.y[0]) / 2;
		//calcul des valeurs de f aux points du milieu des arêtes a,b,c
		double fa = rhsf.f(xa, ya);
		double fb = rhsf.f(xb, yb);
		double fc = rhsf.f(xc, yc);
		// calcul des omega 
		double omega = aire / 3.0;
		// calcul de B_k aux indices coorespondant aux noeuds numérotés s0, s1 et s2
		B[s0] += omega * (0.5 * fa + 0.5 * fc);
		B[s1] += omega * (0.5 * fa + 0.5 * fb);
		B[s2] += omega * (0.5 * fb + 0.5 * fc);
	}

	return B;
}

vector<double> calcAV(const vector<double>& V, int N, int M, double a, double b, double lambda) { // On réutilise matvect sans eta et avec lambda=1
	int G = (N + 1) * (M + 1); // dimension G de Vh
	vector<double> W(G, 0.0); // on crée un vecteur vide qui nous servira pour stocker les AV_k

	lambda = 1; // On met lambda = 1
	vector<double> X = Subdiv(a, N);
	vector<double> Y = Subdiv(b, M);

	vector<vector<int>> maillage = maillageTR(N, M);
	int K = maillage.size();

	for (int t = 0; t < K; t++) { // On parcours chaque triangle
		// sommets du triangle
		int s0 = maillage[t][0];
		int s1 = maillage[t][1];
		int s2 = maillage[t][2];

		// coordonnées des sommets
		vector<int> ij0 = invnumgb(N, M, s0);
		vector<int> ij1 = invnumgb(N, M, s1);
		vector<int> ij2 = invnumgb(N, M, s2);

		// Triangle
		Triangle T(
			X[ij0[0]], X[ij1[0]], X[ij2[0]],
			Y[ij0[1]], Y[ij1[1]], Y[ij2[1]]
		);
		// On calcul les intégrales des lambdas(i), ici ils coincident avec les w(i) puisque l'on est sur le triangle T et qu'on considere des sommets de ce meme triangles
		vector<double> ATReac = ReacTerm({ T.x[0], T.x[1], T.x[2] }, { T.y[0], T.y[1], T.y[2] }, lambda);  // on garde que ReacTerm

		int sommets[3] = { s0, s1, s2 }; // on stock les numéros des sommets

		// Boucle sur les sommets du triangle
		for (int i = 0; i < 3; i++) {
			int k = sommets[i];   // on a le numéro du noeud en question (le i-eme sommet de T)
			double res = 0.0;  // on initialise la somme

			// Boucle sur tous les sommets de T
			for (int j = 0; j < 3; j++) {
				int l = sommets[j]; // numéro du noeud correspondant au j-eme sommet de T
				int idx = indices_sym(i, j); // indice pour faire le lien avec les vecteurs de taille 6 définis dans DiffTerm et ReacTerm
				res += V[l] * ATReac[idx]; // contribution du sommet j par rapport au sommet i
			}

			W[k] += res; // on ajoute la contribution totale de T au noeud k
		}
	}
	return W; // On renvoie W avec les valeurs de AV

}

void VhElement::normL2(int N, int M, double a, double b) // on calcul la norme L2 pour V dans Vhelement
{
	vector<double> W = calcAV(V, N, M, a, b, 1); // Utilsiation de calcAV avec lambda = 1

	double s = 0.0; // On fait le produit scalaire entre V et AV
	for (int k = 0; k < V.size(); k++)
		s += V[k] * W[k];

	L2norm = sqrt(s); // Norme L2
}

void VhElement::normLinfini() { // Calcul de la norme infini comme max des valeurs absolues des coordonées
	int n = V.size();
	double max = fabs(V[0]); // Simple boucle qui itère sur tous les éléments du vecteur et garde le plus grand en v. abs. 
	for (int k = 1; k < n; k++) {
		if (fabs(V[k]) > max) {
			max = fabs(V[k]);
		}
	}
	Linfininorm = max;
}


vector<double> calcAVgrad(const vector<double>& V, int N, int M, double a, double b, ETA& eta_obj) {
	int G = (N + 1) * (M + 1); // dimension G de Vh
	vector<double> W(G, 0.0); // on crée un vecteur vide qui nous servira pour stocker les AV_k

	vector<double> X = Subdiv(a, N);
	vector<double> Y = Subdiv(b, M);

	vector<vector<int>> maillage = maillageTR(N, M);
	int K = maillage.size();

	for (int t = 0; t < K; t++) { // On parcours chaque triangle
		// sommets du triangle
		int s0 = maillage[t][0];
		int s1 = maillage[t][1];
		int s2 = maillage[t][2];

		// coordonnées des sommets
		vector<int> ij0 = invnumgb(N, M, s0);
		vector<int> ij1 = invnumgb(N, M, s1);
		vector<int> ij2 = invnumgb(N, M, s2);

		// Triangle
		Triangle T(
			X[ij0[0]], X[ij1[0]], X[ij2[0]],
			Y[ij0[1]], Y[ij1[1]], Y[ij2[1]]
		);

		vector<double> ATDiff = DiffTerm({ T.x[0], T.x[1], T.x[2] }, { T.y[0], T.y[1], T.y[2] }, eta_obj); // on ne prend que le terme Diff

		int sommets[3] = { s0, s1, s2 }; // on stock les numéros des sommets

		// Boucle sur les sommets du triangle
		for (int i = 0; i < 3; i++) {
			int k = sommets[i];   // on a le numéro du noeud en question (le i-eme sommet de T)
			double res = 0.0;  // on initialise la somme

			// Boucle sur tous les sommets de T
			for (int j = 0; j < 3; j++) {
				int l = sommets[j]; // numéro du noeud correspondant au j-eme sommet de T
				int idx = indices_sym(i, j); // indice pour faire le lien avec les vecteurs de taille 6 définis dans DiffTerm et ReacTerm
				res += V[l] * ATDiff[idx]; // contribution du sommet j par rapport au sommet i
			}

			W[k] += res; // on ajoute la contribution totale de T au noeud k
		}
	}
	return W; // On renvoie W avec les valeurs de AV

}


void VhElement::normL2Grad(int N, int M, double a, double b)
{

	class ETAConstLocal : public ETA {// On crée une sous classe temporaire pour eta=1 uniquement valable pour normL2Grad
	public:
		double eta(double x, double y) { return 1.0; } // eta=1
		double integ_eta(const Triangle& T) {          // intégrale de eta sur un triangle, utilisé dans calcAVgrad
			double detBT = (T.x[1] - T.x[0]) * (T.y[2] - T.y[0]) - (T.x[2] - T.x[0]) * (T.y[1] - T.y[0]);
			return fabs(detBT) / 2.0;
		}
	};

	ETAConstLocal eta_obj;  // eta = 1

	vector<double> W = calcAVgrad(V, N, M, a, b, eta_obj); // calcule AV avec eta=1

	double s = 0.0;
	for (int k = 0; k < V.size(); k++)
		s += V[k] * W[k];

	gradL2norm = sqrt(s); // stocke la norme L2 du gradient
}



vector<double> Gradientbiconj(const vector<double>& B, int N, int M, double a, double b, double lambda, ETA& eta_obj, double epsilon, int maxiter) {
	// On vérifie que les dimensions de B soient cohérentes
	if (B.size() != (N + 1) * (M + 1)) {
		throw invalid_argument("Dimension de B incoherente avec le maillage");
	}
	int n = B.size();
	
	vector<double> X0(n, 0);
	VhElement vh0(n);
	vh0.V = X0;

	// Précalcul du maillage pour gagner du temps de calcul
	vector<double> X = Subdiv(a, N);
	vector<double> Y = Subdiv(b, M);
	vector<vector<int>> maillage = maillageTR(N, M);
	int K = maillage.size(); // Nombre de triangle du maillage

	vector<vector<double>> precomp_diff(K);
	vector<vector<double>> precomp_reac(K);

	// Precalcul des diff term et reac term pour alléger la charge de calcul

	#pragma omp parallel for
	for (int t = 0; t < K ; t++) {
		int s0 = maillage[t][0];
		int s1 = maillage[t][1];
		int s2 = maillage[t][2];

		// coordonnées des sommets
		vector<int> ij0 = invnumgb(N, M, s0);
		vector<int> ij1 = invnumgb(N, M, s1);
		vector<int> ij2 = invnumgb(N, M, s2);

		// Triangle
		Triangle T(
			X[ij0[0]], X[ij1[0]], X[ij2[0]],
			Y[ij0[1]], Y[ij1[1]], Y[ij2[1]]
		);
		// On calcul les intégrales des lambdas(i), ici ils coincident avec les w(i) puisque l'on est sur le triangle T et qu'on considere des sommets de ce meme triangles
		precomp_diff[t] = DiffTerm({T.x[0], T.x[1], T.x[2]}, {T.y[0], T.y[1], T.y[2]}, eta_obj);
		precomp_reac[t] = ReacTerm({T.x[0], T.x[1], T.x[2]}, {T.y[0], T.y[1], T.y[2]}, lambda);
	}


	// Calcul de RO
	vector<double> AX0 = matvec(vh0,X,Y,maillage,precomp_diff,precomp_reac, N, M, lambda);
	vector<double> R0 = SoustrVect(B, AX0);


	//Initialisation des variables avant la boucle
	vector<double> R0_star = R0;
	vector<double> Rj = R0;

	VhElement W0(n);
	W0.V = R0;

	VhElement Wj(n);
	Wj = W0;

	vector<double> Xj = X0;

	double diff = norme_euclidienne(R0);

	int niter = 0;

	//Boucle de Calcul
	while (diff > epsilon && niter<maxiter) {
		//Calcul des alphas_j

		vector<double> AWj = matvec(Wj,X,Y,maillage,precomp_diff,precomp_reac, N, M, lambda);
		double alpha_j = prod_scalaire(Rj, R0_star) / prod_scalaire(AWj,R0_star);

		//Calcul des S_j

		VhElement Sj(n);
		Sj.V = SoustrVect(Rj, prod_par_scal(AWj, alpha_j));

		//Calcul des omega_j

		vector<double> ASj = matvec(Sj,X,Y,maillage,precomp_diff,precomp_reac, N, M, lambda);
		double omega_j = prod_scalaire(ASj, Sj.V)/prod_scalaire(ASj,ASj);

		//Calcul des Xj

		Xj = somme_vect(somme_vect(Xj, prod_par_scal(Wj.V, alpha_j)), prod_par_scal(Sj.V, omega_j));
		
		//Calcul des Rj

		vector<double> Rj_temp = SoustrVect(Sj.V, prod_par_scal(ASj, omega_j));

		//Calcul des Bj

		double temp1 = prod_scalaire(Rj_temp, R0_star);
		double temp2 = prod_scalaire(Rj, R0_star);

		double beta_j = (temp1 / temp2) * (alpha_j / omega_j);

		//Calcul des Wj

		vector<double> u = SoustrVect(Wj.V, prod_par_scal(AWj, omega_j));
		u = prod_par_scal(u, beta_j);
		Wj.V = somme_vect(Rj_temp, u);

		// On remplace Rj temp par Rj

		Rj = Rj_temp;

		//Calcul de la norme du reste

		diff = norme_euclidienne(Rj);

		// fin de la boucle 

		niter++;
	}
	return (Xj);
}

vector<double> interpol(int M, int N, double a, double b, SolExa& solExa) {
	int G = (M + 1) * (N + 1);
	vector<double> U_h(G, 0.0);


	// On génère les subdivisions
	vector<double> X = Subdiv(a, N); 
	vector<double> Y = Subdiv(b, M);

	//Boucle de calcul des éléments du vecteur
	for (int k = 0; k < G; k++) {
		vector<int> id = invnumgb(N, M, k); // On retrouve les coordonées de notre noeud
		double x_k = X[id[0]];
		double y_k = Y[id[1]];
		U_h[k] = solExa.u_p(x_k, y_k); // On calcule les valeurs exactes de up aux noeuds
	}
	return (U_h);
}

vector<double> erreurs(int N, int M, double a, double b, SolExa& solExa, vector<double> u_h) {
	int n = (N + 1) * (M + 1);
	vector<double> vect_err(3, 0.0); // Vecteur destiné à recevoir les différentes erreurs

	VhElement u_hat(n); // On converti u_hat en élément de Vh et on définit tous les paramètres nécessaires à nos calcul
	u_hat.V = interpol(M, N, a, b, solExa); // On définit u_hat comme l'interpolée de notre solution exacte
	u_hat.normL2(N, M, a, b); // On se sert des attributs de notre classe Vh pour calculer les nominateurs des erreurs relatives
	u_hat.normL2Grad(N, M, a, b);
	u_hat.normLinfini();

	VhElement D(n); // On calcule D l'erreur dans Vh puis idem
	D.V = SoustrVect(u_hat.V, u_h);
	D.normL2(N, M, a, b);
	D.normL2Grad(N, M, a, b); 
	D.normLinfini();

	vect_err[0] = D.L2norm / u_hat.L2norm;
	vect_err[1] = D.gradL2norm / u_hat.gradL2norm;
	vect_err[2] = D.Linfininorm / u_hat.Linfininorm;

	return(vect_err);
}

void main() {

	double a = 1.0;
	double b = 1.0;
	double lambda = 1.0;
	double epsilon = 1e-8;
	double eta_0 = 1.0;

	RHSF rhsf(eta_0);
	SolExa solExa;
	ETAEta0 eta(eta_0);

	/*
	ofstream fichier("resultats_convergence.csv");

	for(int k:{5,10,20,50,100,200,500}){
		int N = k;
		int M = N;
		double h = 2.0 / N;
		vector<double> X = Subdiv(a, N);
		vector<double> Y = Subdiv(b, M);

		vector<double> B = scdmembre(N, M, a, b, rhsf);
		vector<double> u_h = Gradientbiconj(B, N, M, a, b, lambda, eta, epsilon, 15000);
		vector<double> results = erreurs(N, M, a, b, solExa, u_h);

		fichier << N << "," << h << "," << results[0] << "," << results[1] << "," << results[2] << endl;
		cout << "N=" << k << " done" << endl;
	}

	*/

	/**/
	int N = 100, M = 100;
	vector<double> X = Subdiv(a, N);
	vector<double> Y = Subdiv(b, M);
	int j_mid = (0.75 * M); // noeud le plus proche de y=0.5

	ofstream fichier_d("resultats_d.csv");

	for (int mv : {100}) {
		m = mv;
		vector<double> B = scdmembre(N, M, a, b, rhsf);
		vector<double> u_h = Gradientbiconj(B, N, M, a, b, lambda, eta, epsilon, 15000);

		// u_h aux points du maillage
		for (int i = 0; i <= N; i++) {
			int k = (N + 1) * j_mid + i;
			fichier_d << mv << "," << X[i] << "," << u_h[k] << ",nan" << endl;
		}

		// u_p sur 1000 points
		for (int i = 0; i < 1000; i++) {
			double x_i = -a + i * (2.0 * a) / 999.0;
			fichier_d << mv << "," << x_i << ",nan," << solExa.u_p(x_i, Y[j_mid]) << endl;
		}

		cout << "Q32d - m=" << mv << " done" << endl;
	}
	fichier_d.close();
	

	/*
	int N = 50;
	int M = N;

	ofstream fichier("erreur_vs_m.csv");

	for (int mv : {1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 42, 44, 46, 48, 50}) {
		m = mv;

		RHSF rhsf(eta_0);  // recréer rhsf car m a changé

		vector<double> B = scdmembre(N, M, a, b, rhsf);
		vector<double> u_h = Gradientbiconj(B, N, M, a, b, lambda, eta, epsilon, 15000);
		vector<double> results = erreurs(N, M, a, b, solExa, u_h);

		fichier << mv << "," << results[0] << "," << results[1] << "," << results[2] << endl;
		cout << "m=" << mv << " done, e0=" << results[0] << endl;
	}
	fichier.close();
	*/
}

