function [str] = my_num2str(num)
    if num>= 0 && num <=9
        str = ['0', num2str(num)];           
    else
        str = num2str(num);
    end
   
end