r1 = [1 2 3 4 5 6]'; 
p = [2 4 5 8 9 7]'; 
r2 = [10 15 1 0 7 9]'; 
for x=0:length(p)
    p=circshift(p,1);
        for w=0:length(r2)
            r2=circshift(r2,1);
            for u =1:1:length(r1)
                v(u)=5*r1(u)^2+2*p(u)^2+r2(u)^2;
            end
              scatter3(r1,p,r2,40,v,'filled')
              xlabel('r1') 
              ylabel('p')
              zlabel('r2')
              hold on
          end
  end
  cb = colorbar;