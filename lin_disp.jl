using MultObjNLPModels, Plots, Random, Colors, ColorSchemes

r(y, ŷ) = sum((y - ŷ).^2)

batch_list = [1,3,5,8,10,16,32,50,64,100]
# batch_list = [1,3,5]
# batch_size = [1,10,50,100]
# data_list = [100,200,500]

K = 100
n = 100

Random.seed!(0)
x = rand(n)
y = 2*x .+ 3 + randn(n) * 0.4
X = [ones(n) x]
nlp = LinearRegressionModel(X,y)

p = plot(size=(600,400),leg=false, xlabel= "Tamanho do batch", ylabel="Resíduos", title="$n dados", xticks=(1:1:10, ["1","3","5","8","10","16","32","50","64","100"]))
# ylims!(p,0.0,18.0)

for nb in batch_list
    res_iter = []
    for k=1:K
        output = sthocastic_gradient(nlp, batch_size=nb)
        β = output.solution
        y_pred = X*β
        res = r(y, y_pred)
        append!(res_iter, res)
    end

    N = findall(batch_list .== nb)
    for num in N
        l = length(res_iter)
        xg = num * ones(l) + randn(l) * 0.1
        scatter!(p, xg, res_iter, opacity=0.6, palette=:jet)
    end
end

png(p, "lineardisp_size_completo$(n)_teste")
