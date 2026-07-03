module load cdo

outdir="/glade/u/home/sressel/spencer-scratch/graphcast_input_data/climatology"
mkdir -p "$outdir"

start_time=$(date +%s)

for year in $(seq 1979 1989); do
    for month in $(seq -w 1 12); do
        ym="${year}${month}"
        echo "$ym"

        # U
        files=$(ls /gdex/data/d633000/e5.oper.an.pl/${ym}/e5.oper.an.pl.*_u*uv*.nc 2>/dev/null)

        if [ -z "$files" ]; then
            echo "No files for $ym"
            continue
        fi

        tmpfiles=()
        for f in $files; do
            tmp="${outdir}/tmp/tmp_$(basename "$f")"
            echo $tmp
            cdo -O -remapbil,grid_1deg.txt -sellevel,200,850 -selhour,0,6,12,18 -selname,U "$f" "$tmp"
            tmpfiles+=("$tmp")
        done

        cdo -O -chname,U,u_component_of_wind -mergetime "${tmpfiles[@]}" "${outdir}/u_${ym}.nc"
        rm -f "${tmpfiles[@]}"

        # V
        files=$(ls /gdex/data/d633000/e5.oper.an.pl/${ym}/e5.oper.an.pl.*_v*uv*.nc 2>/dev/null)

        if [ -z "$files" ]; then
            echo "No files for $ym"
            continue
        fi

        tmpfiles=()
        for f in $files; do
            tmp="${outdir}/tmp/tmp_$(basename "$f")"
            echo $tmp
            cdo -O -remapbil,grid_1deg.txt -sellevel,200 -selhour,0,6,12,18 -selname,V "$f" "$tmp"
            tmpfiles+=("$tmp")
        done

        cdo -O -chname,V,v_component_of_wind -mergetime "${tmpfiles[@]}" "${outdir}/v_${ym}.nc"
        rm -f "${tmpfiles[@]}"
    done
done

end_time=$(date +%s)
time_diff=$((end_time - start_time))
hours=$((time_diff / 3600))
minutes=$(( (time_diff % 3600) / 60 ))
seconds=$((time_diff % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"