function u = benchmark()
global rmat;
global predInd;

Pnum = sum(rmat~=0,1);
Psum = sum(rmat,1);
Pmeans = Psum./Pnum;
u = full(Pmeans(predInd(:,2)))';
end