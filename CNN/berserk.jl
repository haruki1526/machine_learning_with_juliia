using Images
using Distributions
function ReLU(x)
    if(x>0)
        return x
        else
            return 0.0
    end
end
function softmax(u::Array,output_vector_size)
    z=similar(u)
    u_sum=sum(exp.(u-maximum(u)))
    for k in 1:output_vector_size
        z[k]=exp(u[k]-maximum(u))/u_sum
    end
    return z
end


#畳み込み層
function convolution(l)
    x=Int(0)
    y=Int(0)
    for si in 0:S:datay-H
        y+=1
        x=0
        for sj in 0:S:datax-H
            x+=1
            for k in 1:K
                for i in 1:H
                    for j in 1:H
                        u[l,y,x]+=imgdata[k,i+si,j+sj]*h[i,j]+b[l,y,x]
                    end
                end
            end
            z[l,y,x]=ReLU(u[l,y,x])
        end
    end
    return z
end
#チャネルなし　2回め以降
function convolution(l,u::Array, z::Array)
    x=Int(0)
    y=Int(0)
    for si in 0:S:datay-H
        y+=1
        x=0
        for sj in 0:S:datax-H
            x+=1
                for i in 1:H
                    for j in 1:H
                        u[l,y,x]+=z[l-1,i+si,j+sj]*h[i,j]+b[l,y,x]
                    end
                end
            z[l,y,x]=ReLU(u[l,y,x])
        end
    end
    return z
end

function nuwral_net(l,z::Array)
    u=similar(z[l,:])
    u=w[l,:,:]*z[l-1,:]
    z[l,:]=ReLU.(u)
    return z
end

function nuwral_net(l,z::Array,output_vector_size)
    u=similar(z[l,1:output_vector_size])
    u=w[l,1:output_vector_size,:]*z[l-1,:]
    z[l,1:output_vector_size]=softmax(u,output_vector_size)
    return z
end

function maxpooling(l,z::Array)
    x=Int(0)
    y=Int(0)
    for si in 0:S:datay-P
        y+=1
        x=0
        for sj in 0:S:datax-P
            x+=1
            z[l,y,x]=maximum(z[l-1,1+si:P+si,1+sj:P+sj])
        end
    end
    return z
    
end


function dReLU(x)
    if(x>0)
        return 1.0
        else
        return 0.0
    end
end

function cnn(imgdata,datay,datax)

end


function main()
	m=Normal(0,1)
	rand(m,4,4)
	#チャネル数
	K=3 #RGB
	train=[0.0 for i in 1:40]
	train[38]=1.0
	data=load("phot/1berserk.jpg")
	datax=3402 #画像サイズ
	datay=3402
	const Layernum=21
	imgdata=channelview(data) #[RGB,4608,3456]
end



