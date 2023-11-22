using Markdown
using InteractiveUtils
using Images
using ImageMagick
using FFTW
using StatsBase


#       PRIMERA ETAPA
function rellenar(imagen)
    alto_original, ancho_original = size(imagen)
    ancho_nuevo = ceil(Int, ancho_original / 16) * 16
    alto_nuevo = ceil(Int, alto_original / 16) * 16
    
    relleno_ancho = ancho_nuevo - ancho_original
    relleno_alto = alto_nuevo - alto_original
    imagen_nueva = zeros(RGB{N0f8}, alto_nuevo, ancho_nuevo)
    imagen_nueva[1:alto_original, 1:ancho_original] = imagen
    
    return imagen_nueva
end
function conv(mat, ker, stride)
    res = zeros(convert(Tuple{Int, Int}, 1 .+ (size(mat) .- size(ker))./stride))
    for i in 1:(size(res)[1])
        for j in 1:(size(res)[2])
            im = size(ker)[1] + (i-1)*stride
            jm = size(ker)[2] + (j-1)*stride
            res[i,j] = sum(mat[(im - size(ker)[1]+1):(im),
                (jm-size(ker)[2]+1):(jm)].*ker)
        end
    end		
    return res
end

function primera_etapa(image)
    image2 = convert.(YCbCr, rellenar(image))
    canales = channelview(image2)
    
    ker=[1/4 1/4; 1/4 1/4]
    Cb = conv(canales[2,:,:], ker, 2) .- 128
    Cr = conv(canales[3,:,:], ker, 2) .- 128

    return canales[1,:,:] .- 128, Cb, Cr
end
function inv_primera_etapa(Y, Cb, Cr)
    imagen = colorview(YCbCr,
        Y .+ 128,
        repeat(Cb, inner=(2,2)) .+ 128,
        repeat(Cr, inner=(2,2)) .+ 128)
    return RGB.(convert.(RGB, imagen))
end
#       SEGUNDA ETAPA
function transfbloques(M)
    for i in 1:Int(size(M)[1]/8)
        for j in 1:Int(size(M)[2]/8)
            dct!(view(M, (8*i-7):(8*i), (8*j-7):(8*j)))
        end
    end
end
function inv_transfbloques(M)
    for i in 1:Int(size(M)[1]/8)
        for j in 1:Int(size(M)[2]/8)
            idct!(view(M, (8*i-7):(8*i), (8*j-7):(8*j)))
        end
    end
end
Mq = [16 11 10 16 24 40 51 61;
12 12 14 19 26 58 60 55;
14 13 16 24 40 57 69 56;
14 17 22 29 51 87 80 62;
18 22 37 56 68 109 103 77;
24 35 55 64 81 104 113 92;
49 64 78 87 103 121 120 101;
72 92 95 98 112 100 103 99] # No es simetrica
function quant(M, q = Mq)
    for i in 1:Int(size(M)[1]/8)
        for j in 1:Int(size(M)[2]/8)
            M[(8*i-7):(8*i), (8*j-7):(8*j)] = trunc.(Int8, M[(8*i-7):(8*i), (8*j-7):(8*j)] ./ q)
        end
    end
end
function inv_quant(M, q = Mq)
    for i in 1:Int(size(M)[1]/8)
        for j in 1:Int(size(M)[2]/8)
            M[(8*i-7):(8*i), (8*j-7):(8*j)] .*= q
        end
    end
end
function rlezag(M)
    tmp =[1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56,64]
    res = []
    for i in 1:64
        push!(res, M[tmp[i]])
    end
    return rle(res)
end
function inv_rlezag(reps, vals)
    aux = inverse_rle(reps, vals)
    tmp =[1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56,64]
    M = zeros(8, 8)
    for i in 1:64
         M[tmp[i]] = aux[i]
    end
    return M
end
function rlezag_imagen(M)
    bloques = Matrix{Tuple{Vector{Any},Vector{Int64}}}(undef,Int(size(M)[1]/8),Int(size(M)[2]/8))
    for i in 1:Int(size(M)[1]/8)
        for j in 1:Int(size(M)[2]/8)
            bloques[i,j] = rlezag(M[(8*i-7):(8*i), (8*j-7):(8*j)])
        end
    end
    return bloques
end
function inv_rlezag_imagen(bloques)
    M = zeros(Int(size(bloques)[1]*8), Int(size(bloques)[2]*8))
    for i in 1:Int(size(M)[1]/8)
        for j in 1:Int(size(M)[2]/8)
            M[(8*i-7):(8*i), (8*j-7):(8*j)] = inv_rlezag(bloques[i,j]...)
        end
    end
    return M
end

function segunda_etapa(Y, Cb, Cr, q = Mq)
    transfbloques(Y), transfbloques(Cb), transfbloques(Cr)
    quant(Y, q), quant(Cb, q), quant(Cr, q)
    return rlezag_imagen(Y), rlezag_imagen(Cb), rlezag_imagen(Cr)
end
function inv_segunda_etapa(rleY, rleCb, rleCr, q = Mq)
    Y = inv_rlezag_imagen(rleY)
    Cb = inv_rlezag_imagen(rleCb)
    Cr = inv_rlezag_imagen(rleCr)
    inv_quant(Y, q), inv_quant(Cb, q), inv_quant(Cr, q)
    inv_transfbloques(Y), inv_transfbloques(Cb), inv_transfbloques(Cr)
    return Y, Cb, Cr
end

#       EXPORTAR
function grg(img, q = Mq)
    return segunda_etapa(primera_etapa(img)..., q)
end
function inv_grg(rleY, rleCb, rleCr, q = Mq)
    return inv_primera_etapa(inv_segunda_etapa(rleY, rleCb, rleCr, q)...)
end

function toGrg(path, q = Mq)
    #   FUNC. PRINCIPAL
    imagen = load(path)
    imgrg = grg(imagen, q)
    io = open(split(path, ".")[1]*".grg", "w")
    for k in 1:3
        write(io, UInt16(size(imgrg[k])[1]))
        write(io, UInt16(size(imgrg[k])[2]))
        for i in 1:size(imgrg[k])[1]
            for j in 1:size(imgrg[k])[2]
                write(io, Int8(size(imgrg[k][i,j][1])[1]))
                for l in 1:size(imgrg[k][i,j][1])[1]
                    write(io, Int8(imgrg[k][i,j][1][l]))
                    write(io, Int8(imgrg[k][i,j][2][l]))
                end
            end
        end
    end
    for i in 1:8
        for j in 1:8
            write(io, UInt8(q[i, j]))
        end
    end
end
function fromGrg(path)
    #   FUNC. PRINCIPAL
    io = open(path)

    n_G = read(io, UInt16)
    m_G = read(io, UInt16)
    G = Matrix{Tuple{Vector{Any},Vector{Int64}}}(undef,n_G,m_G)
    for i in 1:n_G
        for j in 1:m_G
            a_G = zeros(read(io, Int8))
            b_G = zeros(size(a_G))
            for l in 1:size(a_G)[1]
                a_G[l] = read(io, Int8)
                b_G[l] = read(io, Int8)
            end
            G[i,j] = (a_G, b_G)
        end
    end
    
    n_r = read(io, UInt16)
    m_r = read(io, UInt16)
    r = Matrix{Tuple{Vector{Any},Vector{Int64}}}(undef,n_r,m_r)
    for i in 1:n_r
        for j in 1:m_r
            a_r = zeros(read(io, Int8))
            b_r = zeros(size(a_r))
            for l in 1:size(a_r)[1]
                a_r[l] = read(io, Int8)
                b_r[l] = read(io, Int8)
            end
            r[i,j] = (a_r, b_r)
        end
    end

    n_g = read(io, UInt16)
    m_g = read(io, UInt16)
    g = Matrix{Tuple{Vector{Any},Vector{Int64}}}(undef,n_g,m_g)
    for i in 1:n_g
        for j in 1:m_g
            a_g = zeros(read(io, Int8))
            b_g = zeros(size(a_g))
            for l in 1:size(a_g)[1]
                a_g[l] = read(io, Int8)
                b_g[l] = read(io, Int8)
            end
            g[i,j] = (a_g, b_g)
        end
    end

    q = zeros(8, 8)
    for i in 1:8
        for j in 1:8
            q[i, j] = read(io, UInt8)
        end
    end
    inversa = inv_grg(G, r, g, q)
    save(split(path, ".")[1]*"2.jpg", inversa)
    return inversa
end
