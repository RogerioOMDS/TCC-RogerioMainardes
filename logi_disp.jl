using MultObjNLPModels, Plots, Random

r(y, ŷ) = - sum(y[i] * log(ŷ[i]) + (1 - y[i]) * log(1 - ŷ[i]) for i=1:length(y)) / length(y) 
σ(z) = 1 / (1 + exp(-z))

batch_list = [1,3,5,8,10,16,32,50,64,100]
# batch_list = [1,5,50,100]
# batch_list = [1,3,5]

K = 500
n = 100


p = plot(size=(600,400), xlabel="Resíduos")

Random.seed!(0)
# x = rand(n)
# y = [x + randn() * 0.15 > 0.5 ? 1 : 0 for x in x]
x = rand(n,2)
y = [x[i,1]^2 + x[i,2]^2 > 0.7 + randn() * 0.2 ? 1.0 : 0.0 for i=1:n]
X = [ones(n) x]

nlp = LogisticRegressionModel(X,y)

p = plot(size=(600,400),leg=false, xlabel= "Tamanho do batch", ylabel="Resíduos", title= "$n dados", xticks=(1:1:10, ["1","3","5","8","10","16","32","50","64","100"]))
# ylims!(p, 0.0,4.0)

for nb in batch_list
    res_iter = []
    for k=1:K
        output = sthocastic_gradient(nlp, batch_size=nb)
        β = output.solution
        y_pred = σ.(X*β)
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

png(p, "logistic_disp_completo$(n)_novo3")  

