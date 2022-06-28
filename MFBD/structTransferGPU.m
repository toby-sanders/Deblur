function X = structTransferGPU(X,mode)

% transfer a structure to and from the GPU
% mode = 0 -> send to GPU
% mode = 1 -> gather from GPU

f = fieldnames(X);
for i = 1:numel(f)
    if isnumeric(X.(f{i})) || islogical(X.(f{i})) % transfer numerics only
        if ~mode
            X.(f{i}) = gpuArray(single(X.(f{i})));
        else
            X.(f{i}) = gather(single(X.(f{i})));
        end
    end
end