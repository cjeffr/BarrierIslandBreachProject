module breach_module

    implicit none
    save

    logical, private :: module_setup = .false.

    ! Place data here
    real(kind=8) sigma, data
    real, allocatable :: breach_trigger(:), center(:), south(:), north(:), west(:), east(:) 
    real, allocatable :: depth(:), start_time(:), end_time(:), time_ratio(:)
    integer :: num_breaches, num

contains

    subroutine setup_breaching(path)

        implicit none

        ! Input
        character(len=*), optional, intent(in) :: path

        ! Locals
        integer, parameter :: unit = 54
        character(len=128) :: line


        if (.not.module_setup) then

            ! Open data file
            if (present(path)) then
                open(unit, file = path)
            else
                open(unit, file = 'breach.data')
            endif
	    read(unit, *) num
            num_breaches = num
	    allocate (breach_trigger(num), center(num), south(num), north(num), west(num), east(num))
            allocate (depth(num),  start_time(num), end_time(num), time_ratio(num))
	    print *, num
            ! Basic switch to turn on variable friction
            read(unit, *)
            read(unit, *) breach_trigger
            read(unit, *) south
            read(unit, *) north
            read(unit, *) west
            read(unit, *) east
            read(unit, *) center
            read(unit, *) sigma
            read(unit, *) time_ratio
            read(unit, *) start_time
            read(unit, *) end_time
            read(unit, *) depth
            close(unit)
            module_setup = .true.
        end if

    end subroutine setup_breaching

    subroutine some_other_routine()

        implicit none

        ! Do stuff - maybe call from b4step


    end subroutine some_other_routine


end module breach_module
