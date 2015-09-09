function u = means()
global rmat;
global idmap;

Pnum = sum(rmat~=0,1);
Psum = sum(rmat,1);
Pmeans = Psum./Pnum;
u = full(Pmeans(idmap(:,2)))';
end