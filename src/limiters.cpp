#include "limiters.hpp"
#include <array>
#include <algorithm>

template<typename RealType>
void ComputeGradients(const Triangulation<2,RealType>& tri,
                      ElementData<2,RealType>& elem,
                      std::vector<Tensor<1,2,RealType>>& grad_rho)
{
    const auto& interior = tri.get_interior_faces();
    const auto& boundary = tri.get_boundary_faces();
    const auto& periodic = tri.get_periodic_faces();

    grad_rho.assign(elem.size(), Tensor<1,2,RealType>{0,0});

    // ---------- Interior Faces ----------
    for (unsigned int f = 0; f < interior.size(); ++f)
    {
        unsigned int L = interior.elem_l[f];
        unsigned int R = interior.elem_r[f];

        Tensor<1,2,RealType> n = {
            interior.normal_x[f],
            interior.normal_y[f]
        };

        RealType length = interior.face_area[f];

        RealType u_hat =
            0.5 * (elem.density[L] + elem.density[R]);

        Tensor<1,2,RealType> contrib = u_hat * n * length;

        grad_rho[L] = grad_rho[L] + contrib;
        grad_rho[R] = grad_rho[R] - contrib;
    }

    // ---------- Boundary Faces ----------
    for (unsigned int f = 0; f < boundary.size(); ++f)
    {
        unsigned int e = boundary.elem[f];

        Tensor<1,2,RealType> n = {
            boundary.normal_x[f],
            boundary.normal_y[f]
        };

        RealType length = boundary.face_area[f];

        // Use interior state or BC state
        RealType u_hat = elem.density[e];

        grad_rho[e] = grad_rho[e] + u_hat * n * length;
    }

    // ---------- Periodic Faces ----------
    for (unsigned int f = 0; f < periodic.size(); ++f)
    {
        unsigned int L = periodic.elem_l[f];
        unsigned int R = periodic.elem_r[f];

        Tensor<1,2,RealType> n = {
            periodic.normal_x[f],
            periodic.normal_y[f]
        };

        RealType length = periodic.face_area[f];

        RealType u_hat =
            0.5 * (elem.density[L] + elem.density[R]);

        Tensor<1,2,RealType> contrib = u_hat * n * length;

        grad_rho[L] = grad_rho[L] + contrib;
        grad_rho[R] = grad_rho[R] - contrib;
    }

    // ---------- Divide by Area ----------
    for (unsigned int i = 0; i < elem.size(); ++i)
        grad_rho[i] = grad_rho[i] * elem.inv_area[i];
}

Tensor<2,4,double> 
Limiter_BJ(Tensor<2,4, double>& L0,
           Tensor<1,4,double>& u0,
           std::array<Tensor<1,2,double>,3>& r) {

    //define the tensor product between the gradient and the vector to the cell nodes
    Tensor<1,4,double> Lr1 = {r[0][0] * L0(0,0) + r[0][1] * L0(1,0), 
                              r[0][0] * L0(0,1), + r[0][1] * L0(1,1), 
                              r[0][0] * L0(0,2) + r[0][1] * L0(1,2), 
                              r[0][0] * L0(0,3) + r[0][1] * L0(1,3)};

    Tensor<1,4,double> Lr2 = {r[1][0] * L0(0,0) + r[1][1] * L0(1,0),
                              r[1][0] * L0(0,1) + r[1][1] * L0(1,1), 
                              r[1][0] * L0(0,2) + r[1][1] * L0(1,2),
                              r[1][0] * L0(0,3) + r[1][1] * L0(1,3)};

    Tensor<1,4,double> Lr3 = {r[2][0] * L0(0,0) + r[2][1] * L0(1,0),
                              r[2][0] * L0(0,1) + r[2][1] * L0(1,1), 
                              r[2][0] * L0(0,2) + r[2][1] * L0(1,2), 
                              r[2][0] * L0(0,3) + r[2][1] * L0(1,3)};

    //get adjacent states
    Tensor<1,4,double> u1 = u0 + Lr1;
    Tensor<1,4,double> u2 = u0 + Lr2;
    Tensor<1,4,double> u3 = u0 + Lr3;
    std::vector<Tensor<1,4,double>> ui = {u1, u2, u3};

    //find umin and umax based on adjacent states
    Tensor<1,4,double> umin;
    Tensor<1,4,double> umax;
    //loop through the state
    for (int j = 0; j < 4; j++) {
        umin[j] = std::min({u0[j], u1[j], u2[j], u3[j]});
        umax[j] = std::max({u0[j], u1[j], u2[j], u3[j]});
    }

    //find the scalar limiter
    Tensor<1,4,double> alpha = {1,1,1,1};
    //loop through the nodes of an element
    for (int i = 0; i < 3; i++) {
        double alpha_cmp;
        Tensor<1,4,double> uiN = ui[i];
        
        //loop through the states
        for (int j = 0; j < 4; j++) {

            //compute the required scalar limiter for each node
            if (uiN[j] - u0[j] > 0.0) {
                alpha_cmp = std::min(1.0, (umax[j] - u0[j])/(uiN[j] - u0[j]));
            } else if (uiN[j] - u0[j] < 0.0) {
                alpha_cmp = std::min(1.0, (umin[j] - u0[j])/(uiN[j] - u0[j]));
            } else {
                alpha_cmp = 1.0;
            }//end if

            //set scalar limiter to the minimum of the adjacent nodes
            alpha[j] = std::min(alpha[j], alpha_cmp);
        }//end state for
    }//end node for


    //return the limited gradient
    Tensor<2,4, double> limited = L0;
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            limited(j,i) = alpha[j] * L0(j,i);
        }
    }
    return limited;
}//end Limiter_BJ

template<typename RealType>
std::array<Tensor<1,2,RealType>,3>
ComputeVertexVectors(unsigned int elem_id,
                     const MeshData& data,
                     const ElementData<2,RealType>& elem)
{
    unsigned int v0 = data.node_1[elem_id];
    unsigned int v1 = data.node_2[elem_id];
    unsigned int v2 = data.node_3[elem_id];

    RealType cx = elem.centroid_x[elem_id];
    RealType cy = elem.centroid_y[elem_id];

    Tensor<1,2,RealType> r0 = { data.x[v0] - cx,
                                data.y[v0] - cy };

    Tensor<1,2,RealType> r1 = { data.x[v1] - cx,
                                data.y[v1] - cy };

    Tensor<1,2,RealType> r2 = { data.x[v2] - cx,
                                data.y[v2] - cy };

    return { r0, r1, r2 };
} //end ComputeVertexVectors

