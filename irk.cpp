#include <iostream>
#include <string>

#include "irk.hpp"


namespace irk {



/**
   \brief expands the coefficient lists.

   This is needed to automatically calculate the interpolating coefficients.

   This is like operator expansion of operator( c1, c2 )
   if c1 = { a1 + a2 } and c2 = { b1 + b2 }
   and we encode for that as c1 = { {1}, {2} }; c2 = { {3}, {4} }
   then the expansion would be operator(c1,c2) =
   { a1b1 + a1b2 + a2b1 + a2b2 } which would be encoded as
   { {1,3}, {1,4}, {2,3}, {2,4} }.
   operator( ( a1b1 + a1b2 + a2b1 + a2b2 ), (x1 + x2) ) follows from induction.
*/
typedef std::vector<std::vector<int> > coeff_list;

coeff_list expand( const coeff_list &c1,
                   const coeff_list &c2 )
{
	// Test driven:
	// 1. expand( { {1}, {2} }, { {3}, {4} } ) should lead to
	// { {1,3}, {1,4}, {2,3}, {2,4} }
	//
	// 2. expand( { {1,3}, {1,4}, {2,3}, {2,4} }, { {5}, {6} } ) leads to
	// { {1,3,5}, {1,4,5}, {2,3,5}, {2,4,5},
	//   {1,3,6}, {1,4,6}, {2,3,6}, {2,4,6} }
	//
	// 3. expand( { {1,2,3}, {4,5} } ) should be
	// ( a1 + a2 + a3 ) ( b1 + b2 ) =
	// ( a1b1 + a2b1 + a3b1 + a1b2 + a2b2 + a3b2 ) =
	// { {1,4}, {2,4}, {3,4}, {1,5}, {2,5}, {2,6} }
	//
	coeff_list c3;

	for( std::size_t i = 0; i < c1.size(); ++i ){
		for( std::size_t j = 0; j < c2.size(); ++j ){
			std::vector<int> vij( c1[i].begin(), c1[i].end() );
			vij.insert( vij.end(), c2[j].begin(), c2[j].end() );
			c3.push_back( vij );
		}
	}
	return c3;
}


/**
   \brief Output operator for a coefficient list.
*/
std::ostream &operator<<( std::ostream &o, const coeff_list &c )
{
	o << " { ";
	for( std::size_t i = 0; i < c.size(); ++i ){
		o << "{";
		for( std::size_t j = 0; j < c[i].size(); ++j ){
			o << " " << c[i][j];
		}
		o << " }";
		if( i < c.size() - 1 ) o << ", ";
	}
	o << " }";
	return o;
}


/**
   \brief Simple calculation of the factorial of n.
*/
int factorial(int n)
{
	if( n <= 1 ) return 1;

	int res = 1;
	for( int i = 1; i <= n; ++i ){
		res *= i;
	}

	return res;
}

/**
   \brief binomial coefficient (n above b).
*/
int binom_coeff( int n, int b )
{
	return factorial(n) / (factorial(b)*factorial(n-b));
}



bool verify_solver_coeffs( const solver_coeffs &sc )
{
	auto N = sc.b.size();
	if( N != sc.c.size() || N != sc.A.n_rows || N != sc.A.n_cols ){
		return false;
	}
	if( N == 0 ) return false;

	return true;
}


mat_type collocation_interpolate_coeffs( const vec_type& c )
{
	// Interpolates on a solution interval as
	// b_j(t) = b_interp(j,0)*t + b_interp(j,1)*t^2
	//          + b_interp(j,2)*t^3 + ...
	// The coefficients arise from the following equation:
	//
	// bj(t) = integral( lp_j(x), dx ),
	// where lp_j(x) = (x-c2)(x-c1)(x-c3) / (cj-c2)(cj-c1)(cj-c3)
	//
	// Therefore, to derive the shape of the polynomial, we have
	// to perform some expansion in terms of all the coefficients.
	//
	// For example, if we have three stages, then we have
	//
	// lp_1(x) = (x - c2)(x - c3)/(c1-c2)(c1-c3)
	// lp_2(x) = (x - c1)(x - c3)/(c2-c1)(c2-c3)
	// lp_3(x) = (x - c1)(x - c2)/(c3-c1)(c3-c2)
	//
	// The denominators are easily dealt with. For the nominators, we
	// need to expand the products. We encode x with -1, c1 with 0, etc.
	// Therefore, for three stages we should get...
	//
	// expand( { {-1}, {1} }, { {-1}, {2} } ) =
	// { {-1 -1}, {-1 2}, {-1 1}, {1 2} }.
	// This means x^2  + c2 x + c1 x + c1c2
	//
	// Afterwards, we integrate in x, so for this case, we'd obtain
	// x^3/3  + c2 x^2 / 2 + c1 x ^ 2 / 2 + c1c2 * x
	//

	std::size_t Ns = c.size();
	vec_type d(Ns);

	for( std::size_t i = 0; i < Ns; ++i ){
		double cfacs = 1.0;
		for( std::size_t j = 0; j < Ns; ++j ){
			if( j == i ) continue;
			cfacs *= c(i) - c(j);
		}
		d(i) = cfacs;
	}

	std::vector<coeff_list> poly_coefficients(Ns);
	for( std::size_t i = 0; i < Ns; ++i ){
		coeff_list ci;
		for( std::size_t j = 0; j < Ns; ++j ){
			if( i == j ) continue;

			int jj = j;
			// -1 codes for x.
			if( ci.size() == 0 ){
				ci = {{-1}, {jj}};
			}else{
				coeff_list tmp = expand( ci, { {-1}, {jj} } );
				ci = tmp;
			}
		}
		poly_coefficients[i] = ci;
	}

	// The coefficients construct the polynomial, so from this, we can
	// exactly calculate the polynomial coefficients. It is convoluted
	// but it works...
	//
	// The number of -1 s encode for the power of the term, with
	// four -1 s meaning it is a fifth order term, etc.
	mat_type b_interp = arma::zeros( Ns, Ns );

	for( std::size_t i = 0; i < Ns; ++i ){
		for( const std::vector<int> &cf : poly_coefficients[i] ){
			// Check the order of the term.
			int order = 0;
			for( int j : cf ){
				if( j == -1 ) order++;
			}
			// order0 is the constant term that will become linear.
			// If you grab the highest order term, ignore it.
			// if( order == static_cast<int>(Ns) ) continue;

			double coeff = 1.0 / d(i) / (1.0 + order);
			for( int j : cf ){
				if( j != -1 ){
					// Multiply by the right c:
					coeff *= -c[j];
				}
			}

			int b_interp_idx = Ns - order - 1;
			int Ns_nosign = Ns;
			assert( (b_interp_idx >= 0) && "index out of range" );
			assert( (b_interp_idx < Ns_nosign) &&
			        "index out of range" );

			b_interp(i, b_interp_idx) += coeff;
		}
	}

	return b_interp;
}




solver_coeffs get_coefficients( int method )
{
	solver_coeffs sc;
	double sqrt3 = sqrt(3.0);
	// double sqrt5 = sqrt(5.0);
	double sqrt6 = sqrt(6.0);
	double sqrt21 = sqrt(21.0);
	// double sqrt15 = sqrt(15.0);

	// Methods that need adding:
	// LOBATTO_IIIA_85,
	// LOBATTO_IIIC_127,
	// GAUSS_LEGENDRE_{63,105}

	sc.FSAL = false;
	sc.name = method_to_name( method );
	sc.gamma = 0.0;

	switch(method){
	default:
		std::cerr << "Method " << method << " not supported!\n";
		break;

	case IMPLICIT_EULER:
		sc.A = { 1.0 };
		sc.b = { 1.0 };
		sc.c = { 1.0 };
		sc.order = 1;
		sc.order2 = 0;
		break;

	case LOBATTO_IIIA_43:

		sc.A = { {      0.0,     0.0,       0.0 },
		         { 5.0/24.0, 1.0/3.0, -1.0/24.0 },
		         {  1.0/6.0, 2.0/3.0,  1.0/6.0 } };

		sc.c = { 0.0, 0.5, 1.0 };
		sc.b = { 1.0/6.0, 2.0/3.0, 1.0/6.0 };
		sc.b2 = { -0.5, 2.0, -0.5 };
		sc.order = 4;
		sc.order2 = 2;
		sc.FSAL = true;
		break;


	case LOBATTO_IIIC_43:

		sc.A = { { 1.0/6.0, -1.0/3.0, 1.0/6.0 },
		         { 1.0/6.0, 5.0/12.0, -1.0/12.0 },
		         { 1.0/6.0, 2.0/3.0, 1.0/6.0 } };
		sc.b  = { 1.0/6.0, 2.0/3.0, 1.0/6.0 };
		sc.b2 = { -0.5, 2.0, -0.5 };  // Need a better set because these are not L-stable
		sc.c  = { 0.0, 0.5, 1.0 };
		sc.order = 4;
		sc.order2 = 2;

		break;

	case LOBATTO_IIIC_85:
		sc.A = { {1.0/20.0,
		          -7.0/60.0,
		          2.0/15.0,
		          -7.0/60.0,
		          1.0/20.0},

		         {1.0/20.0,
		          29.0/180.0,
		          (47.0  - 15*sqrt21)/315.0,
		          (203.0 - 30*sqrt21)/1260.0,
		          -3.0/140.0},

		         {1.0/20.0,
		          (329.0 + 105.0*sqrt21)/2880.0,
		          73.0/360.0,
		          (329.0 - 105.0*sqrt21)/2880.0,
		          3.0/160.0},

		         {1.0/20.0,
		          (203.0 + 30*sqrt21)/1260.0,
		          (47+15*sqrt21)/315.0,
		          29/180.0,
		          -3/140.0},

		         {1.0/20.0,
		          49/180.0,
		          16/45.0,
		          49/180.0,
		          1.0/20.0}};
		sc.c  = {0.0, 0.5 - sqrt21/14.0, 0.5, 0.5 + sqrt21/14.0, 1.0};
		sc.b  = {0.05, 49/180.0, 16/45.0, 49/180.0, 0.05};
		sc.b2 = {-0.3640805339798258,
		         0.9561694999656932,
		         0.3214814814814818,
		         -0.1576509814471765,
		         0.05458333333333142};
		sc.gamma  = 0.18949720064649575035;
		sc.order  = 8;
		sc.order2 = 5;

		sc.b_interp = collocation_interpolate_coeffs( sc.c );

		break;
	case GAUSS_LEGENDRE_42:

		sc.A = { { 0.25, 0.25 - sqrt3/6.0, 0.0 },
		         { 0.25 + sqrt3/6.0, 0.25, 0.0 },
		         { 0.0, 0.0, 0.0 } };
		sc.c = { 0.5 - sqrt3/6.0, 0.5 + sqrt3/6.0, 0.0 };
		sc.b = { 0.5, 0.5, 0.0 };
		sc.b2= { (3*sqrt3 + 1)/12.0, (7-sqrt3)/12.0, (2-sqrt3)/6.0 };
		sc.order = 4;
		sc.order2 = 2;


		break;

	case RADAU_IIA_32:{

		sc.A = { {5.0/12.0, -1.0/12.0},
		         {3.0/4.0, 1.0/4.0 } };

		sc.gamma = 1.0/3.0;

		sc.c  = { 1.0/3.0, 1.0 };
		sc.b  = { 3.0/4.0, 1.0/4.0 };

		sc.b2 = { (-6*sc.gamma + 3.0) / 4.0,
		          ( 2*sc.gamma + 1.0) / 4.0 };

		sc.order = 3;
		sc.order2 = 2;

		sc.b_interp = collocation_interpolate_coeffs( sc.c );

		break;
	}


	case RADAU_IIA_53:{

		// Recalculate these with the Mathematica script.

		sc.A = { { (88 - 7*sqrt6)/360.0, (296 - 169*sqrt6)/1800.0, (-2+3*sqrt6)/225.0 },
		         { (296 + 169*sqrt6)/1800.0, (88 + 7*sqrt6)/360.0, (-2-3*sqrt6)/225.0 },
		         { (16.0 - sqrt6)/36.0, (16 + sqrt6)/36.0, 1.0 / 9.0 } };
		// gamma is the real eigenvalue of A.
		sc.gamma = 2.74888829595677e-01;


		sc.c  = {  (4.0-sqrt6)/10.0,  (4.0+sqrt6) / 10.0, 1.0 };
		sc.b  = {  (16 - sqrt6)/36.0,
		           (16 + sqrt6)/36.0,
		           1.0 / 9.0 };

		sc.b2 = { -((18*sqrt6 + 12)*sc.gamma - 16 + sqrt6)/36.0,
		          ((18*sqrt6 - 12)*sc.gamma + 16 + sqrt6)/36.0,
		          -(3*sc.gamma - 1) / 9.0 };

		sc.order = 5;
		sc.order2 = 3;

		sc.b_interp = collocation_interpolate_coeffs( sc.c );



		break;
	}

	case RADAU_IIA_95:{
		sc.A = {{ 0.0729988643179033243, -0.0267353311079455719,
		          0.0186769297639843544, -0.0128791060933064399,
		          0.00504283923388201521 },
		        { 0.153775231479182469, 0.146214867847493507,
		          -0.036444568905128090, 0.021233063119304719,
		          -0.007935579902728778 },
		        { 0.14006304568480987, 0.29896712949128348,
		          0.16758507013524896, -0.03396910168661775,
		          0.01094428874419225 },
		        { 0.14489430810953476, 0.2765000687601592,
		          0.3257979229104210, 0.1287567532549098,
		          -0.01570891737880533 },
		        { 0.1437135607912259, 0.2813560151494621,
		          0.3118265229757413, 0.2231039010835707,
		          0.04 } };

		sc.c = { 0.05710419611451768219312119255411562124,
		         0.27684301363812382768004599768562514112,
		         0.58359043236891682005669766866291724869,
		         0.86024013565621944784791291887511976674,
		         1.0 };

		sc.b = { 0.1437135607912259,
		         0.2813560151494621,
		         0.3118265229757413,
		         0.2231039010835707,
		         0.04 };

		// gamma is the real eigenvalue of A.
		sc.gamma = 0.1590658444274690;

		sc.b2 = { sc.b(0) - 1.5864079001863282*sc.gamma,
		          sc.b(1) + 1.0081178814983730*sc.gamma,
		          sc.b(2) - 0.73097486615978746*sc.gamma,
		          sc.b(3) + 0.50926488484774272*sc.gamma,
		          sc.b(4) - 0.2*sc.gamma };

		sc.order  = 9;
		sc.order2 = 5;

		sc.b_interp = collocation_interpolate_coeffs( sc.c );

		break;
	}

	case RADAU_IIA_137: {

		sc.A = { {  0.037546264993921331333686127624105551410,
		            -0.014039334556460401537626568603936253927,
		            0.010352789600742300936755479003273124789,
		            -0.008158322540275011909204543577213278349,
		            0.006388413879534684943755951486405680409,
		            -0.004602326779148655499352025854768521774,
		            0.001828942561470643704035856835298607816 },

		         {  0.080147596515618967795215595316188773479,
		            0.081062063985891536679584719357221980975,
		            -0.021237992120711034937085469604419103837,
		            0.014000291238817118983742204835134926788,
		            -0.010234185730090163829199816607636044634,
		            0.007153465151364590498062382166962141175,
		            -0.002812639372406723340342762967473461717 },

		         {  0.072063846941881902113362526561137596780,
		            0.171068354983886619424352504009050302800,
		            0.109614564040072109233220407461845690823,
		            -0.024619871728984053862318864441100561089,
		            0.014760377043950817073195348981742706482,
		            -0.009575259396791400556328724726641713431,
		            0.003672678397138305671569774234741682832 },

		         {  0.075705125819824420424641229496338921970,
		            0.154090155142171144646331682046482915172,
		            0.227107736673202386411281287949366350098,
		            0.117478187037024781987912680673932161442,
		            -0.023810827153044173582047929325774334376,
		            0.012709985533661205633610757619788395065,
		            -0.004608844281289633440336366654612469297 },

		         {  0.073912342163191846540806321243016399213,
		            0.161355607615942432186220145903094810374,
		            0.206867241552104197819578846437670730910,
		            0.237007115342694234762246772957327514747,
		            0.103086793533813446624105845745721640646,
		            -0.018854139152580448840052190417863035125,
		            0.005858900974888791823977618246677391072 },

		         {  0.074705562059796230172292559361766628756,
		            0.158307223872468700658479384514628716574,
		            0.214153423267200031108697457856861396619,
		            0.219877847031860039987487355490766771106,
		            0.198752121680635269801826469184534504760,
		            0.069265501605509133230972165761976742365,
		            -0.008116008197728290107881426350852749124 },

		         {  0.074494235556010317933248780209166920975,
		            0.159102115733650740872435217234934182108,
		            0.212351889502977804199154019575104122356,
		            0.223554914507283234749674476821221017986,
		            0.190474936822115576902969173938062761867,
		            0.119613744612656202893538740384776300830,
		            0.020408163265306122448979591836734693878 }
		};

		sc.b = { 0.074494235556010317933248780209166920975,
		         0.159102115733650740872435217234934182108,
		         0.212351889502977804199154019575104122356,
		         0.223554914507283234749674476821221017986,
		         0.190474936822115576902969173938062761867,
		         0.119613744612656202893538740384776300830,
		         0.020408163265306122448979591836734693878 };

		sc.c = { 0.029316427159784891972050276913164910374,
		         0.148078599668484291849976852495979212230,
		         0.336984690281154299097052972080775705198,
		         0.558671518771550132081393341805521940074,
		         0.769233862030054500916883360115645451837,
		         0.926945671319741114851873965819682011056,
		         1 };


		sc.gamma =  0.111896465300035075935905337180769194745;


		sc.b2 = { sc.b(0) - 1.59406421856104180 * sc.gamma,
		          sc.b(1) + 1.03655375219647650 * sc.gamma,
		          sc.b(2) - 0.79382172349079269 * sc.gamma,
		          sc.b(3) + 0.63257765224993423 * sc.gamma,
		          sc.b(4) - 0.49761071360300131 * sc.gamma,
		          sc.b(5) + 0.35922239406556795 * sc.gamma,
		          sc.b(6) - 0.14285714285714286 * sc.gamma };

		sc.order  = 13;
		sc.order2 = 7;

		sc.b_interp = collocation_interpolate_coeffs( sc.c );


		break;
	}

	case GAUSS_LEGENDRE_147: {

		sc.A = { {  0.0323712415422174233,
		            -0.0114510172831838703,
		            0.00763320387242354493,
		            -0.00513373356322534498,
		            0.00317505877368563764,
		            -0.00160681903704610586,
		            0.000458109523749452982 },
		         {  0.0700435413787260763,
		            0.0699263478723191670,
		            -0.0165900065788477712,
		            0.00934962278344333208,
		            -0.00539709193189613785,
		            0.00264584386673003738,
		            -0.000743850190171923617 },
		         {  0.0621539357873498645,
		            0.152005522057830993,
		            0.0954575126262797362,
		            -0.0183752442154518374,
		            0.00871256259847518198,
		            -0.00395358015881043812,
		            0.00107671561562791674 },
		         {  0.0663329286176847006,
		            0.133595769223882288,
		            0.207701880765970782,
		            0.104489795918367347,
		            -0.0167868555134113096,
		            0.00625692652075604553,
		            -0.00159044553324985392 },
		         {  0.0636657674688069299,
		            0.143806275903448772,
		            0.182202462654084290,
		            0.227354836052186531,
		            0.0954575126262797362,
		            -0.0121528263131926586,
		            0.00258854729708498210 },
		         {  0.0654863332746067703,
		            0.137206851877908297,
		            0.196312117184455610,
		            0.199629969053291362,
		            0.207505031831407244,
		            0.0699263478723191670,
		            -0.00530105829429122964},
		         { 0.0642843735606853937,
		           0.141459514781684440,
		           0.187739966478873835,
		           0.214113325399960039,
		           0.183281821380135928,
		           0.151303713027822204,
		           0.0323712415422174233 } };
		sc.c = { 0.0254460438286207377,
		         0.129234407200302780,
		         0.297077424311301417,
		         0.5,
		         0.702922575688698583,
		         0.870765592799697220,
		         0.974553956171379262 };

		sc.b = { 0.0647424830844348466,
		         0.139852695744638334,
		         0.190915025252559472,
		         0.208979591836734694,
		         0.190915025252559472,
		         0.139852695744638334,
		         0.0647424830844348466 };

		sc.gamma = 0.100567464822504836;

		sc.b2 = { sc.b(0) - 1.57466249971055050 * sc.gamma,
		          sc.b(1) + 0.97072669650612219 * sc.gamma,
		          sc.b(2) - 0.67210786192236179 * sc.gamma,
		          sc.b(3) + 0.45714285714285714 * sc.gamma,
		          sc.b(4) - 0.28405414676522997 * sc.gamma,
		          sc.b(5) + 0.14407010361206885 * sc.gamma,
		          sc.b(6) - 0.04111514886290593 * sc.gamma  };

		break;
	}

	case LOBATTO_IIIA_127: {

		sc.A = { {0, 0, 0, 0, 0, 0, 0},
		         {0.0328462643282926479, 0.0593228940275514045,
		          -0.0107685944511892671, 0.00559759178056977723,
		          -0.00348892997080746277, 0.00221709658891453970,
		          -0.000838270442615104380},
		         {0.0180022232018151657,
		          0.157701130641689042, 0.102354812046861915,
		          -0.0184782592734590440, 0.00957758010074140595,
		          -0.00568186456622437757, 0.00209998111321878573},
		         {0.0275297619047619048,
		          0.127788255559837470, 0.237485652721645444,
		          0.121904761904761905, -0.0216129621167141318,
		          0.0106247681209455044, -0.00372023809523809524},
		         {0.0217095426963050238, 0.144094888247007352,
		          0.206295110504189906, 0.262287783082982854,
		          0.113517878558069396, -0.0192881069609060680,
		          0.00580730060770864382},
		         {0.0246477942521389139,
		          0.136195927091868434, 0.219361620575738774,
		          0.238211932028954032, 0.226641285056120579,
		          0.0790901296532315695, -0.00903674051876883836},
		         {0.0238095238095238095, 0.138413023680782974,
		          0.215872690604931312, 0.243809523809523810,
		          0.215872690604931312, 0.138413023680782974,
		          0.0238095238095238095 } };

		sc.c = { 0,
		         0.0848880518607165351,
		         0.265575603264642893,
		         0.5,
		         0.734424396735357107,
		         0.915111948139283465,
		         1 };

		sc.b = { 0.0238095238095238095,
		         0.138413023680782974,
		         0.215872690604931312,
		         0.243809523809523810,
		         0.215872690604931312,
		         0.138413023680782974,
		         0.0238095238095238095 };

		sc.gamma = 0.5;

		sc.b2 = { sc.b(0) - sc.gamma,
		          sc.b(1),
		          sc.b(2),
		          sc.b(3),
		          sc.b(4),
		          sc.b(5),
		          sc.b(6) };

		break;
	}

	}

	// Some checks:
	for( std::size_t i = 0; i < sc.c.size(); ++i ){
		double ci = sc.c(i);
		double si = 0.0;
		for( std::size_t j = 0; j < sc.c.size(); ++j ){
			si += sc.A(i,j);
		}
		if( std::fabs( si - ci ) > 1e-5 ){
			std::cerr << "Warning! Mismatch between c and A(i,:) "
			          << "for i = " << i << ", method = "
			          << method_to_name( method ) << "!\n";
		}
	}

	return sc;
}


solver_options default_solver_options()
{
	solver_options s;
	return s;
}



bool verify_solver_options( solver_options &opts )
{
	if( opts.newton_opts ) return true;
	std::cerr << "ERROR! solver_opts @" << &opts << " does not have "
	          << "newton::options set!\n";
	return false;
}


const char *method_to_name( int method )
{
	return irk::rk_method_to_string[method].c_str();
}


int name_to_method( const std::string &name )
{
	return irk::rk_string_to_method[name];
}


std::vector<std::string> all_method_names()
{
	std::vector<std::string> methods;
	for( auto pair : rk_string_to_method ){
		methods.push_back( pair.first );
	}
	return methods;
}


vec_type project_b( double theta, const irk::solver_coeffs &sc )
{
	assert( sc.b_interp.size() > 0 && "Chosen method does not have dense output!" );

	std::size_t Ns = sc.b.size();
        vec_type ts(Ns);

	// ts will contain { t, t^2, t^3, ..., t^{Ns} }
	double tt = theta;
	for( std::size_t i = 0; i < Ns; ++i ){
		int j = Ns - i - 1;
		ts(j) = tt;
		tt *= theta;
	}
	// Now bs = sc.b_interp * ts;
        return sc.b_interp * ts;
}


rk_output merge_rk_output( const rk_output &sol1, const rk_output &sol2 )
{
	rk_output merger( sol1 );
	merger.status |= sol2.status;

	merger.t_vals.insert( merger.t_vals.end(),
	                      sol2.t_vals.begin(), sol2.t_vals.end() );
	merger.y_vals.insert( merger.y_vals.end(),
	                      sol2.y_vals.begin(), sol2.y_vals.end() );
	merger.stages.insert( merger.stages.end(),
	                      sol2.stages.begin(), sol2.stages.end() );
	merger.err_est.insert( merger.err_est.end(),
	                       sol2.err_est.begin(), sol2.err_est.end() );
	merger.err.insert( merger.err.end(),
	                   sol2.err.begin(), sol2.err.end() );

	merger.elapsed_time += sol2.elapsed_time;
	double steps1 = sol1.t_vals.size();
	double steps2 = sol2.t_vals.size();
	double total_steps = steps1 + steps2;
	merger.accept_frac = steps1*sol1.accept_frac + steps2*sol2.accept_frac;
	merger.accept_frac /= total_steps;

	merger.count.attempt += sol2.count.attempt;
	merger.count.reject_newton += sol2.count.reject_newton;
	merger.count.reject_err += sol2.count.reject_err;
	merger.count.newton_success += sol2.count.newton_success;
	merger.count.newton_incr_diverge += sol2.count.newton_incr_diverge;
	merger.count.newton_iter_error_too_large +=
		sol2.count.newton_iter_error_too_large;
	merger.count.newton_maxit_exceed += sol2.count.newton_maxit_exceed;

	return merger;
}



void restore_state( const rk_output &sol, double &t, vec_type &y, vec_type &K,
                    vec_type &err_est, double errs[3], double dts[3] )
{
	// Attempt to restore state from an rk_output object.

	// sol already stores
	//  - Last time value
	//  - Last solution
	//  - Last stage values
	//  - Last error estimate vector
	//  - Last error
	// need:
	//  - dt (can be calculated).
	//


}

} // namespace irk
