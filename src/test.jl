using LinearAlgebra
using ITensors

function NumericalGradientZ(mpo::MPO;accuracy=1e-8)
    grad_r = []
    grad_i = []
    for j in 1:N
        push!(grad_r,zeros(ComplexF64,size(mpo[j])))
        push!(grad_i,zeros(ComplexF64,size(mpo[j])))
    end
    
    epsilon = zeros(ComplexF64,size(mpo[1]));
    # Site 1
    for i in 1:length(epsilon)
        epsilon[i] = accuracy
        eps = ITensor(epsilon,inds(mpo[1]))
        #eps = ITensor(epsilon,s_i[1],s_o[1],link[1]);
        mpo[1] += eps
        loss_p = log(Normalization(mpo))
        mpo[1] -= eps
        loss_m = log(Normalization(mpo))
        grad_r[1][i] = (loss_p-loss_m)/(accuracy)
        
        epsilon[i] = im*accuracy
        eps = ITensor(epsilon,inds(mpo[1]))
        #eps = ITensor(epsilon,s_i[1],s_o[1],link[1]);
        mpo[1] += eps
        loss_p = log(Normalization(mpo))
        mpo[1] -= eps
        loss_m = log(Normalization(mpo))
        grad_i[1][i] = (loss_p-loss_m)/(im*accuracy)
        
        epsilon[i] = 0.0
    end

    for j in 2:N-1
        epsilon = zeros(ComplexF64,size(mpo[j]));
        for i in 1:length(epsilon)
            epsilon[i] = accuracy
            eps = ITensor(epsilon,inds(mpo[j]))
            #eps = ITensor(epsilon,link[j-1],s_i[j],s_o[j],link[j]);
            mpo[j] += eps
            loss_p = log(Normalization(mpo))
            mpo[j] -= eps
            loss_m = log(Normalization(mpo))
            grad_r[j][i] = (loss_p-loss_m)/(accuracy)
            
            epsilon[i] = im*accuracy
            eps = ITensor(epsilon,inds(mpo[j]))
            #eps = ITensor(epsilon,link[j-1],s_i[j],s_o[j],link[j]);
            mpo[j] += eps
            loss_p = log(Normalization(mpo))
            mpo[j] -= eps
            loss_m = log(Normalization(mpo))
            grad_i[j][i] = (loss_p-loss_m)/(im*accuracy)

            epsilon[i] = 0.0
        end
    end

    # Site N
    epsilon = zeros(ComplexF64,size(mpo[N]));
    for i in 1:length(epsilon)
        epsilon[i] = accuracy
        eps = ITensor(epsilon,inds(mpo[N]))
        #eps = ITensor(epsilon,link[N-1],s_i[N],s_o[N]);
        mpo[N] += eps
        loss_p = log(Normalization(mpo))
        mpo[N] -= eps
        loss_m = log(Normalization(mpo))
        grad_r[N][i] = (loss_p-loss_m)/(accuracy)

        epsilon[i] = im*accuracy
        eps = ITensor(epsilon,inds(mpo[N]))
        #eps = ITensor(epsilon,link[N-1],s_i[N],s_o[N]);
        mpo[N] += eps
        loss_p = log(Normalization(mpo))
        mpo[N] -= eps
        loss_m = log(Normalization(mpo))
        grad_i[N][i] = (loss_p-loss_m)/(im*accuracy)
        
        epsilon[i] = 0.0
    end
    return grad_r-grad_i

end;

function CheckGradients(grads,num_grads)
  print("\n\n")
  println("\033[95m\033[1m","Testing Gradients","\033[0m\n")
  # Site=1
  ind_g = inds(grads[1])
  counter = 1
  for l in 1:dim(ind_g[3])
    for s2 in 1:2
      for s1 in 1:2
        grad = grads[1][ind_g[1]=>s1,ind_g[2]=>s2,ind_g[3]=>l]
        num_grad = num_grads[1][counter]
        print(grad," \t ",num_grad," \t ")
        if (abs(grad-num_grad)<1e-6)
          println("\033[92m\033[1m","PASSED","\033[0m")
        else
          println("\033[91m\033[1m","FAILED","\033[0m")
        end 
        counter += 1
      end
    end
  end
  print("\n\n")
  for j in 2:N-1
    ind_g = inds(grads[j])
    counter = 1
    for l2 in 1:dim(ind_g[4])
      for s2 in 1:2
        for s1 in 1:2
          for l1 in 1:dim(ind_g[1])
            grad = grads[j][ind_g[1]=>l1,ind_g[2]=>s1,ind_g[3]=>s2,ind_g[4]=>l2]
            num_grad = num_grads[j][counter]
            print(grad," \t ",num_grad," \t ")
            if (abs(grad-num_grad)<1e-6)
              println("\033[92m\033[1m","PASSED","\033[0m")
            else
              println("\033[91m\033[1m","FAILED","\033[0m")
            end 
            counter += 1
          end
        end
      end
    end
  end
  print("\n\n")
  # Site=N
  ind_g = inds(grads[N])
  counter = 1
  for s2 in 1:2
    for s1 in 1:2
      for l in 1:dim(ind_g[1])
        grad = grads[N][ind_g[1]=>l,ind_g[2]=>s1,ind_g[3]=>s2]
        num_grad = num_grads[N][counter]
        print(grad," \t ",num_grad," \t ")
        if (abs(grad-num_grad)<1e-6)
          println("\033[92m\033[1m","PASSED","\033[0m")
        else
          println("\033[91m\033[1m","FAILED","\033[0m")
        end 
        counter += 1
      end
    end
  end
  print("\n\n")

end


