#include "../include/vort.h"

//@todo Implement this
namespace Vtx {

    Vortex::Vortex():uid(suid++){ }
    Vortex::Vortex(int2 coords, double2 coordsD, int winding, int isOn, std::size_t timeStep):uid(suid++){
        this->coords = coords;
        this->coordsD = coordsD;
        this->winding = winding;
        this->isOn = isOn;
        this->timeStep = timeStep;
    }
    Vortex::~Vortex(){ }

    void Vortex::updateUID(std::size_t uid){
        this->uid = uid;
    }
    void Vortex::updateWinding(int winding){
        this->winding = winding;
    }
    void Vortex::updateIsOn(bool isOn){
        this->isOn = isOn;
    }
    void Vortex::updateCoords(int2 coords){
        this->coords = coords;
    }
    void Vortex::updateCoordsD(double2 coordsD){
        this->coordsD = coordsD;
    }
    void Vortex::updateTimeStep(std::size_t timeStep){
        this->timeStep = timeStep;
    }

    std::size_t Vortex::getUID(){
        return this->uid;
    }
    int Vortex::getWinding(){
        return this->winding;
    }
    bool Vortex::getIsOn(){
        return this->isOn;
    }
    int2 Vortex::getCoords(){
        return this->coords;
    }
    double2 Vortex::getCoordsD(){
        return this->coordsD;
    }
    std::size_t Vortex::getTimeStep(){
        return this->timeStep;
    }

//######################################################################################################################
//######################################################################################################################

    VtxList::VtxList() { }
    VtxList::VtxList(std::size_t reserveSize) {
        vortices.reserve(reserveSize);
    }
    VtxList::~VtxList() {
        this->getVortices().clear();
    }

    //Add vortex to end of list.
    void VtxList::addVtx(std::shared_ptr<Vtx::Vortex> vtx) {
        this->getVortices().push_back(vtx);
    }
    //Add vortex to list at the given idx
    void VtxList::addVtx(std::shared_ptr<Vtx::Vortex> vtx, std::size_t idx) {
        this->getVortices().insert(this->vortices.begin() + idx, vtx);
    }

    //Will return nullptr if idx is outside the range. Otherwise, removes element at idx, and returns a shared_ptr to it
    std::shared_ptr<Vtx::Vortex> VtxList::removeVtx(size_t idx) {
        std::shared_ptr<Vtx::Vortex> v = nullptr;
        if(idx < this->vortices.size()){
             v = this->vortices[idx];
            this->vortices.erase(VtxList::vortices.begin() + idx);
        }
        return v;
    }

    std::vector<std::shared_ptr<Vtx::Vortex> >& VtxList::getVortices(){
        return this->vortices;
    }

    //Assumes UID exists.
    std::shared_ptr<Vtx::Vortex> VtxList::getVtx_Uid(std::size_t uid){
        for(auto a : this->vortices){
            if(a->getUID() != uid){
                continue;
            }
            else
                return a;
        }
        return nullptr;
    }

    std::shared_ptr<Vtx::Vortex> VtxList::getVtx_Idx(std::size_t idx) {
        return this->vortices[idx];
    }

    //Assumes UID exists
    std::size_t VtxList::getVtxIdx_Uid(std::size_t uid){
        for(std::size_t t = 0; t < this->vortices.size(); ++t){
            if(this->vortices[t]->getUID() != uid){
                continue;
            }
            else
                return t;
        }
    }

    //Decremented by 1 as value is post-incremented, and suid will always be 1 higher than largest stored value
    std::size_t VtxList::getMax_Uid(){
        return Vtx::Vortex::suid -1;
    }

    //Compare the distances between vtx and the vortex list. Used for time-based tracking
    std::shared_ptr<Vtx::Vortex> VtxList::getVtxMinDist(std::shared_ptr<Vortex> vtx){
        double dist = std::numeric_limits<double>::max(), distTmp=0.; // Start large
        double2 pos0 = vtx->getCoordsD(), pos1;
        std::size_t idx=0;
        for(std::size_t i=0; i < this->vortices.size(); ++i){
            pos1 = this->vortices[i]->getCoordsD();
            distTmp = sqrt(pow(pos0.x-pos1.x,2) + pow(pos0.y - pos1.y,2));
            if( dist > distTmp && distTmp > 0){
                dist = distTmp;
                idx = i;
            }
        }
        return this->vortices[idx];
    }

    void VtxList::swapUid(std::shared_ptr<Vtx::Vortex> v1, std::shared_ptr<Vtx::Vortex> v2){
        std::size_t uid1 = v1->getUID();
        v1->updateUID(v2->getUID());
        v2->updateUID(uid1);
    }

    void VtxList::swapUid_Idx(std::size_t idx0, std::size_t idx1){
        std::size_t uid0 = this->vortices[idx0]->getUID();
        this->vortices[idx0]->updateUID(this->vortices[idx1]->getUID());
        this->vortices[idx1]->updateUID(uid0);
    }


    void VtxList::sortVtxUID(){
        sort(this->getVortices().begin(), this->getVortices().end(),
             [](std::shared_ptr<Vtx::Vortex> v0, std::shared_ptr<Vtx::Vortex> v1)
                     ->
                     bool{
                 return (v0->getUID() < v1->getUID());
             }
        );
    }
}

