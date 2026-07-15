start_time=$(date +%s)

module load cdo
export HDF5_DISABLE_ERROR_STACK=1

logfile="data_download_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$logfile") 2>&1

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $*"
}

start_year=2001
end_year=2010   # matches your end_date="2010-01-01" -> years loop covers 2001-2009 in the python range;
                # extract_years_months likely stops before end_date, adjust if needed

pressure_level_base="/gdex/data/d633000/e5.oper.an.pl"
outdir="/glade/u/home/sressel/spencer-scratch/graphcast_input_data/daily_climatology"
target_grid="r360x181"

mkdir -p "$outdir"

declare -A var_short=( [u_component_of_wind]="u" [v_component_of_wind]="v" )
declare -A var_old_name=( [u_component_of_wind]="U" [v_component_of_wind]="V" )

for year in $(seq $start_year $end_year); do
    log "$year"

    yeardir="${outdir}/${year}"
    mkdir -p "$yeardir"

    for variable in u_component_of_wind; do
        short="${var_short[$variable]}"
        oldname="${var_old_name[$variable]}"

        log "-- $variable"

        outfile="${yeardir}/${variable}.nc"
        if [ -f "$outfile" ]; then
            log "---- Skipping $year/$variable (already exists)"
            continue
        fi

        pattern="${pressure_level_base}/${year}*/e5.oper.an.pl.*_${short}.*.nc"
        files=$(ls $pattern 2>/dev/null)

        if [ -z "$files" ]; then
            log "No files found for $variable in $year"
            continue
        fi

        # v only needs level 200 (matches the python .sel(level=200) for v)
        if [ "$variable" == "v_component_of_wind" ]; then
            levels="200"
        else
            levels="200,850"
        fi

        chain=()
        for f in $files; do
            chain+=(-remapbil,${target_grid} -sellevel,${levels} -selhour,0 -selname,${oldname} "$f")
        done

        t0=$(date +%s)
        cdo -O -chname,${oldname},${variable} -mergetime "${chain[@]}" "$outfile"
        t1=$(date +%s)
        log "Completed $year/$variable in $((t1 - t0))s"
    done
done

end_time=$(date +%s)
time_diff=$((end_time - start_time))
hours=$((time_diff / 3600))
minutes=$(( (time_diff % 3600) / 60 ))
seconds=$((time_diff % 60))
log "Total time: ${hours}h ${minutes}m ${seconds}s"