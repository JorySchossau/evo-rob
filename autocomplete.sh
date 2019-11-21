
## source this file to add shell autocomplete for the sci tool

_sciautocomp()
{
    local cur prev
    cur=${COMP_WORDS[COMP_CWORD]}
    prev=${COMP_WORDS[COMP_CWORD-1]}
    case ${COMP_CWORD} in
        1)
            COMPREPLY=($(compgen -W "evolve test info analyze" -- ${cur}))
            ;;
        2)
            case ${prev} in
                evolve)
                    COMPREPLY=($(compgen -W "-N -K -G --dev-update --rate --gen --screen-update --lod-update --hist-gens --mup --muv --seed --use-point --use-col-mu --use-row-mu --local-mu --save" -- ${cur}))
                    ;;
                test)
                    COMPREPLY=($(compgen -W "-N -K -G --dev-update --rate --gen --screen-update --lod-update --hist-gens --mup --muv --seed --use-point --use-col-mu --use-row-mu --local-mu --save" -- ${cur}))
                    ;;
                analyze)
                    COMPREPLY=($(compgen -W "--save --pick --ntrials --maxmutations --rate --fitness --phenotype --robustness" -- ${cur}))
                    ;;
            esac
            ;;
        *)
            COMPREPLY=()
            ;;
    esac
}
complete -F _sciautocomp ./sci
