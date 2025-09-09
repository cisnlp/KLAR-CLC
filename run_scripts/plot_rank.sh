# Activate virtual environment
source activate llmtt

# Plot rank curves averaged across all relations
python plot_scripts/rank_plot_en_pivot.py \
    --results_dir llm-transparency-tool-bak/YOUR_RESULTS_DIR \
    --plot_individual_relations False \
    --plot_average_all_relations True \
    --y_type rank

for relation in applies_to_jurisdiction capital capital_of continent \
                country_of_citizenship developer field_of_work \
                headquarters_location instrument language_of_work_or_name \
                languages_spoken location_of_formation manufacturer \
                native_language occupation official_language \
                owned_by place_of_birth place_of_death religion
do
    for lang in en ca es fr hu ja ko nl ru uk vi zh
    do  

        # Plot rank curves for each factual instance and the average
        python plot_scripts/rank_plot_en_pivot.py \
            --results_dir llm-transparency-tool-bak/YOUR_RESULTS_DIR \
            --relation $relation \
            --languages $lang \
            --plot_individual_relations True \
            --plot_average_all_relations False \
            --y_type rank
    done
done