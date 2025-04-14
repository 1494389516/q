%function F=add_dots(F)
% subprogram,remove and add back dots uniformly 
%F=strrep(strrep(strrep(F,'.*','*'),'./','/'),'.^','^');
%F=strrep(strrep(strrep(F,'*','.*'),'/','./'),'^','.^');



function F_str = add_dots(F)
% 输入F可以是符号表达式或字符串，输出为带点操作符的字符串
    if isa(F, 'sym')
        F_str = char(F);   % 将符号表达式转换为字符数组
    else
        F_str = F;         % 若已经是字符串则直接使用
    end
    
    % 移除所有现有点运算符（可选步骤，根据需求调整）
    F_str = strrep(F_str, '.*', '*');
    F_str = strrep(F_str, './', '/');
    F_str = strrep(F_str, '.^', '^');
    
    % 添加点运算符以实现元素级运算
    F_str = strrep(F_str, '*', '.*');
    F_str = strrep(F_str, '/', './');
    F_str = strrep(F_str, '^', '.^');
end