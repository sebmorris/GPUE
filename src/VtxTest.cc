//
// Created by Lee James O'Riordan on 3/1/17.
//

#define NUM_VORT 3

#include "../include/vort.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <algorithm>
#include <cassert>
#include <random>

int main(){
    //########################################################################//

    int2 c = {1,2};
    double2 cD = {3.1,4.2};

    std::mt19937 rng(time(0));
    std::uniform_real_distribution<> dist(0,1);

    //VtxList::VtxList()
    std::shared_ptr<Vtx::VtxList> vl = std::make_shared<Vtx::VtxList>(11);
    std::shared_ptr<Vtx::VtxList> vlp = std::make_shared<Vtx::VtxList>(11);

    //########################################################################//
    // Vortex::Vortex(int2 coords, double2 coordsD, int winding, int isOn, std::size_t timeStep)
    //########################################################################//

    //Create UID -1 vortex, and check values
    std::shared_ptr<Vtx::Vortex> vtx = std::make_shared<Vtx::Vortex>(c,cD, 1, true, 0);

    assert( vtx->getUID() == -1 );
    assert( vtx->getCoords().x == 1 );
    assert( vtx->getCoordsD().y == 4.2 );
    assert( vtx->getWinding() == 1 );
    assert( vtx->getIsOn() == true );
    assert( vtx->getTimeStep() == 0 );
    std::cout << "Vortex value initialisation check passed"<<std::endl;

    //########################################################################//
    // VtxList::addVtx(std::shared_ptr<Vtx::Vortex> vtx)
    // VtxList::getVortices()
    //########################################################################//

    assert(vl->getVortices().size() == 0);
    vl->addVtx(vtx);
    assert(vl->getVortices().size() == 1);
    std::cout << "Vortex add to VtxList check passed"<<std::endl;

    //########################################################################//
    // VtxList::getVtx_Idx(std::size_t idx)
    //########################################################################//

    //Create 0-9 UID vortices
    for(int i =0; i < 10; ++i){
        c.x = i;
        c.y = i;
        cD.x =  i+dist(rng);
        cD.y =  i+dist(rng);
        vl->addVtx(std::make_shared<Vtx::Vortex>(c, cD, 1, true, 0));
    }

    for (int i = 0; i < vl->getVortices().size(); ++i){
        assert(vl->getVtx_Idx(i)->getUID() == i);
    }
    std::cout << "Passed linear UID check passed"<<std::endl;

    //########################################################################//
    // VtxList::getVtxIdx_Uid(std::size_t uid)
    // VtxList::removeVtx(size_t idx)
    //########################################################################//

    vl->removeVtx(2);
    assert(vl->getVortices().size() == 10);
    std::cout << "Passed remove UID 2 check passed"<<std::endl;

    //########################################################################//
    //
    //########################################################################//
    c.x++; c.y++; cD.x += 1.; cD.y += 1.;
    vl->addVtx(std::make_shared<Vtx::Vortex>(c,cD, -1, false, 1000));
    assert(vl->getVtx_Uid(11)->getIsOn() == false);
    assert(vl->getVortices().size() == 11);
    std::cout << "Passed add new vortex with uid=suid++ (11), and getVtx_Uid"<<std::endl;

    //########################################################################//
    //########################################################################//

    assert(vl->getVtx_Idx(10)->getIsOn() == false);
    assert(vl->getVtx_Idx(10)->getWinding() == -1);
    assert(vl->getVtx_Idx(10)->getWinding() == vl->getVtx_Uid(11)->getWinding());
    std::cout << "Passed getVtx_Idx "<<std::endl;

    //########################################################################//
    // VtxList::addVtx(std::shared_ptr<Vtx::Vortex> vtx, std::size_t idx)
    //########################################################################//

    vl->addVtx(std::make_shared<Vtx::Vortex>(c,cD, 3, false, 1000),5);
    assert(vl->getVtx_Idx(5)->getIsOn() == false);
    assert(vl->getVtx_Idx(5)->getWinding() == 3);
    assert(vl->getVtxIdx_Uid(12) == 5);
    assert(vl->getVtx_Idx(11)->getWinding() == vl->getVtx_Uid(11)->getWinding());
    std::cout << "Passed addVtx at arbitrary position "<<std::endl;

    //########################################################################//
    // VtxList::getMax_Uid()
    //########################################################################//
    //Value is post-incremented, so will always be +1 higher than largest UID stored

    assert(vl->getMax_Uid() == 12);
    std::cout << "Passed max UID check "<<std::endl;

    //########################################################################//
    // swapUid(std::shared_ptr<Vtx::Vortex> v1, std::shared_ptr<Vtx::Vortex> v2)
    //########################################################################//
    assert(vl->getVtx_Uid(6)->getUID() == vl->getVtx_Idx(6)->getUID());
    vl->swapUid(vl->getVtx_Uid(6),vl->getVtx_Uid(10));
    assert(vl->getVtx_Uid(10)->getUID() == vl->getVtx_Idx(6)->getUID());
    std::cout << "Passed swap UID check 1"<<std::endl;

    //########################################################################//
    // swapUid_Idx(std::size_t idx0, std::size_t idx1)
    //########################################################################//
    assert(vl->getVtx_Uid(6)->getUID() == vl->getVtx_Idx(10)->getUID());
    vl->swapUid_Idx(vl->getVtxIdx_Uid(6),vl->getVtxIdx_Uid(10));
    assert(vl->getVtx_Uid(6)->getUID() == vl->getVtx_Idx(6)->getUID());
    std::cout << "Passed swap UID check 2"<<std::endl;

    //########################################################################//
    // VtxList::getVtxMinDist(std::shared_ptr<Vortex> vtx)
    //########################################################################//
    assert(vl->getVtxMinDist(vl->getVtx_Uid(8))->getUID() == 9 || vl->getVtxMinDist(vl->getVtx_Uid(8))->getUID() == 7);
    std::cout << "Passed getVtxMinDist check (even if it isn't full rigorous) "<<std::endl;

    //########################################################################//
    // VtxList::sortVortices()
    //########################################################################//
    assert( vl->getVtx_Idx(6)->getUID() == 6 );
    vl->sortVtxUID();
    assert( vl->getVtx_Idx(6)->getUID() == 7 );
    std::cout << "Passed sortVortices check"<<std::endl;

    //########################################################################//
    // VtxList::arrangeVtx(std::vector<std::shared_ptr<Vtx::Vortex> > &vPrev)
    //########################################################################//
    //Vtx::Vortex::resetSUID();
    for( int i = 0; i < 11; ++i ){
        c.x = i;
        c.y = i;
        cD.x =  i+dist(rng);
        cD.y =  i+dist(rng);
        vlp->addVtx(std::make_shared<Vtx::Vortex>(c, cD, 1, true, 0));
    }
    vl->arrangeVtx(vlp->getVortices());

    //########################################################################//
    // VtxList::minDistPair(std::shared_ptr<Vortex> vtx, double minRange)
    //########################################################################//
    for (int i=0; i < 11; ++i){
        auto e = vl->minDistPair(vlp->getVtx_Idx(i), 5);
        if(e.second != nullptr)
            std::cout << "UIDc=" << e.second->getUID() << " UIDp=" << i << "\n\n";
        else
            1;
    }
    std::cout << std::endl;
//nvcc -g -G ./vort.cc -o vort.o -std=c++11 -c -Wno-deprecated-gpu-targets; nvcc -g -G ./VtxTest.cc -std=c++11 -o VtxTest.o -c -Wno-deprecated-gpu-targets; nvcc vort.o VtxTest.o -Wno-deprecated-gpu-targets; ./a.out
    return 0;
}
